import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions
def load_signer_lib(signer_lib_path):
    _LOGGER.debug('loading signer library from %s', signer_lib_path)
    lib = ctypes.CDLL(signer_lib_path, winmode=0) if sys.version_info >= (3, 8) and os.name == 'nt' else ctypes.CDLL(signer_lib_path)
    lib.GetCertPemForPython.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.GetCertPemForPython.restype = ctypes.c_int
    lib.SignForPython.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.SignForPython.restype = ctypes.c_int
    return lib