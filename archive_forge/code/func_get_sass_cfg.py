from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def get_sass_cfg(self, cc=None):
    return disassemble_cubin_for_cfg(self.get_cubin(cc=cc))