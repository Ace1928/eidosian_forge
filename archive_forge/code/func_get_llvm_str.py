from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def get_llvm_str(self):
    return '\n\n'.join(self.llvm_strs)