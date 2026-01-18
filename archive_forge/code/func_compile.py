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
def compile(self, **options):
    """Perform Compilation.

        Compilation options are accepted as keyword arguments, with the
        following considerations:

        - Underscores (`_`) in option names are converted to dashes (`-`), to
          match NVVM's option name format.
        - Options that take a value will be emitted in the form
          "-<name>=<value>".
        - Booleans passed as option values will be converted to integers.
        - Options which take no value (such as `-gen-lto`) should have a value
          of `None` passed in and will be emitted in the form "-<name>".

        For documentation on NVVM compilation options, see the CUDA Toolkit
        Documentation:

        https://docs.nvidia.com/cuda/libnvvm-api/index.html#_CPPv418nvvmCompileProgram11nvvmProgramiPPKc
        """

    def stringify_option(k, v):
        k = k.replace('_', '-')
        if v is None:
            return f'-{k}'
        if isinstance(v, bool):
            v = int(v)
        return f'-{k}={v}'
    options = [stringify_option(k, v) for k, v in options.items()]
    c_opts = (c_char_p * len(options))(*[c_char_p(x.encode('utf8')) for x in options])
    err = self.driver.nvvmVerifyProgram(self._handle, len(options), c_opts)
    self._try_error(err, 'Failed to verify\n')
    err = self.driver.nvvmCompileProgram(self._handle, len(options), c_opts)
    self._try_error(err, 'Failed to compile\n')
    reslen = c_size_t()
    err = self.driver.nvvmGetCompiledResultSize(self._handle, byref(reslen))
    self._try_error(err, 'Failed to get size of compiled result.')
    ptxbuf = (c_char * reslen.value)()
    err = self.driver.nvvmGetCompiledResult(self._handle, ptxbuf)
    self._try_error(err, 'Failed to get compiled result.')
    self.log = self.get_log()
    if self.log:
        warnings.warn(self.log, category=NvvmWarning)
    return ptxbuf[:]