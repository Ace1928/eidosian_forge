from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def run_nvdisasm(cubin, flags):
    fd = None
    fname = None
    try:
        fd, fname = tempfile.mkstemp()
        with open(fname, 'wb') as f:
            f.write(cubin)
        try:
            cp = subprocess.run(['nvdisasm', *flags, fname], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            msg = 'nvdisasm has not been found. You may need to install the CUDA toolkit and ensure that it is available on your PATH.\n'
            raise RuntimeError(msg) from e
        return cp.stdout.decode('utf-8')
    finally:
        if fd is not None:
            os.close(fd)
        if fname is not None:
            os.unlink(fname)