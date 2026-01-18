import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def lazy_add_module(self, buffer):
    """
        Lazily add an NVVM IR module to a compilation unit.
        The buffer should contain NVVM module IR either in the bitcode
        representation or in the text representation.
        """
    err = self.driver.nvvmLazyAddModuleToProgram(self._handle, buffer, len(buffer), None)
    self.driver.check_error(err, 'Failed to add module')