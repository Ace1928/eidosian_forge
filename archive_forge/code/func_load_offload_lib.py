import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions
def load_offload_lib(offload_lib_path):
    _LOGGER.debug('loading offload library from %s', offload_lib_path)
    lib = ctypes.CDLL(offload_lib_path, winmode=0) if sys.version_info >= (3, 8) and os.name == 'nt' else ctypes.CDLL(offload_lib_path)
    lib.ConfigureSslContext.argtypes = [SIGN_CALLBACK_CTYPE, ctypes.c_char_p, ctypes.c_void_p]
    lib.ConfigureSslContext.restype = ctypes.c_int
    return lib