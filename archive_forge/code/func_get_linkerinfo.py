from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def get_linkerinfo(self, cc):
    try:
        return self._linkerinfo_cache[cc]
    except KeyError:
        raise KeyError(f'No linkerinfo for CC {cc}')