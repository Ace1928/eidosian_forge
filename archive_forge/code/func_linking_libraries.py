from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
@property
def linking_libraries(self):
    libs = []
    for lib in self._linking_libraries:
        libs.extend(lib.linking_libraries)
        libs.append(lib)
    return libs