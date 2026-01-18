from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def add_linking_library(self, library):
    library._ensure_finalized()
    self._raise_if_finalized()
    self._linking_libraries.add(library)