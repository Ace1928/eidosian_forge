from llvmlite import ir
from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm, runtime
from numba.cuda.cudadrv.libs import get_cudalib
import os
import subprocess
import tempfile
def get_cufunc(self):
    if self._entry_name is None:
        msg = 'Missing entry_name - are you trying to get the cufunc for a device function?'
        raise RuntimeError(msg)
    ctx = devices.get_context()
    device = ctx.device
    cufunc = self._cufunc_cache.get(device.id, None)
    if cufunc:
        return cufunc
    cubin = self.get_cubin(cc=device.compute_capability)
    module = ctx.create_module_image(cubin)
    cufunc = module.get_function(self._entry_name)
    self._cufunc_cache[device.id] = cufunc
    return cufunc