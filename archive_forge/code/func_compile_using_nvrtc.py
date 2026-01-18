import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
def compile_using_nvrtc(source, options=(), arch=None, filename='kern.cu', name_expressions=None, log_stream=None, cache_in_memory=False, jitify=False):

    def _compile(source, options, cu_path, name_expressions, log_stream, jitify):
        if jitify:
            options, headers, include_names = _jitify_prep(source, options, cu_path)
        else:
            headers = include_names = ()
            major_version, minor_version = _get_nvrtc_version()
            if major_version >= 12:
                options += ('--device-as-default-execution-space',)
        if not runtime.is_hip:
            arch_opt, method = _get_arch_for_options_for_nvrtc(arch)
            options += (arch_opt,)
        else:
            method = 'ptx'
        prog = _NVRTCProgram(source, cu_path, headers, include_names, name_expressions=name_expressions, method=method)
        try:
            compiled_obj, mapping = prog.compile(options, log_stream)
        except CompileException as e:
            dump = _get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                e.dump(sys.stderr)
            raise
        return (compiled_obj, mapping)
    if not cache_in_memory:
        with tempfile.TemporaryDirectory() as root_dir:
            cu_path = os.path.join(root_dir, filename)
            with open(cu_path, 'w') as cu_file:
                cu_file.write(source)
            return _compile(source, options, cu_path, name_expressions, log_stream, jitify)
    else:
        cu_path = '' if not jitify else filename
        return _compile(source, options, cu_path, name_expressions, log_stream, jitify)