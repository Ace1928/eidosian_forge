import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128
def init_ool():
    """
    Same as init() for OOL mode.
    """
    global ffi_ool, cffi_sin_ool, cffi_cos_ool, cffi_foo, cffi_bool_ool
    global vsSin, vdSin, vector_real, vector_imag
    if ffi_ool is None:
        ffi_ool, mod = load_ool_module()
        cffi_sin_ool = mod.lib.sin
        cffi_cos_ool = mod.lib.cos
        cffi_foo = mod.lib.foo
        cffi_bool_ool = mod.lib.boolean
        vsSin = mod.lib.vsSin
        vdSin = mod.lib.vdSin
        vector_real = mod.lib.vector_real
        vector_imag = mod.lib.vector_imag
        del mod