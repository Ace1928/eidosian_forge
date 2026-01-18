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
def compile_using_nvcc(source, options=(), arch=None, filename='kern.cu', code_type='cubin', separate_compilation=False, log_stream=None):
    from cupy.cuda import get_nvcc_path
    if not arch:
        arch = _get_arch()
    if code_type not in ('cubin', 'ptx'):
        raise ValueError('Invalid code_type %s. Should be cubin or ptx')
    if code_type == 'ptx':
        assert not separate_compilation
    arch_str = '-gencode=arch=compute_{cc},code=sm_{cc}'.format(cc=arch)
    _nvcc = get_nvcc_path()
    cmd = _nvcc.split()
    cmd.append(arch_str)
    with tempfile.TemporaryDirectory() as root_dir:
        first_part = filename.split('.')[0]
        path = os.path.join(root_dir, first_part)
        cu_path = '%s.cu' % path
        result_path = '%s.%s' % (path, code_type)
        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)
        if not separate_compilation:
            cmd.append('--%s' % code_type)
            cmd += list(options)
            cmd.append(cu_path)
            try:
                _run_cc(cmd, root_dir, 'nvcc', log_stream)
            except NVCCException as e:
                cex = CompileException(str(e), source, cu_path, options, 'nvcc')
                dump = _get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
                if dump:
                    cex.dump(sys.stderr)
                raise cex
        else:
            cmd_partial = cmd.copy()
            cmd_partial.append('--cubin')
            obj = path + '.o'
            cmd += list(options + ('-o', obj))
            cmd.append(cu_path)
            try:
                _run_cc(cmd, root_dir, 'nvcc', log_stream)
            except NVCCException as e:
                cex = CompileException(str(e), source, cu_path, options, 'nvcc')
                dump = _get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
                if dump:
                    cex.dump(sys.stderr)
                raise cex
            options = _remove_rdc_option(options)
            options += ('--device-link', obj, '-o', path + '.cubin')
            cmd = cmd_partial + list(options)
            try:
                _run_cc(cmd, root_dir, 'nvcc', log_stream)
            except NVCCException as e:
                cex = CompileException(str(e), '', '', options, 'nvcc')
                raise cex
        if code_type == 'ptx':
            with open(result_path, 'rb') as ptx_file:
                return ptx_file.read()
        elif code_type == 'cubin':
            with open(result_path, 'rb') as bin_file:
                return bin_file.read()
        else:
            assert False, code_type