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
def compile_using_hipcc(source, options, arch, log_stream=None):
    cmd = ['hipcc', '--genco'] + list(options)
    with tempfile.TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        in_path = path + '.cpp'
        out_path = path + '.hsaco'
        with open(in_path, 'w') as f:
            f.write(source)
        cmd += [in_path, '-o', out_path]
        try:
            output = _run_cc(cmd, root_dir, 'hipcc', log_stream)
        except HIPCCException as e:
            cex = CompileException(str(e), source, in_path, options, 'hipcc')
            dump = _get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                cex.dump(sys.stderr)
            raise cex
        if not os.path.isfile(out_path):
            raise HIPCCException('`hipcc` command does not generate output file. \ncommand: {0}\nstdout/stderr: \n{1}'.format(cmd, output))
        with open(out_path, 'rb') as f:
            return f.read()