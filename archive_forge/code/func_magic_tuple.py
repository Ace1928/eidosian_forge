from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def magic_tuple(self):
    """
        Return a tuple unambiguously describing the codegen behaviour.
        """
    ctx = devices.get_context()
    cc = ctx.device.compute_capability
    return (runtime.runtime.get_version(), cc)