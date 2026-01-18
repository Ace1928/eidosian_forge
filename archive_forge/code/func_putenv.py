import os
import six
import sys
from ctypes import cdll
from ctypes import CFUNCTYPE
from ctypes import CDLL
from ctypes import POINTER
from ctypes import Structure
from ctypes import byref
from ctypes import cast
from ctypes import sizeof
from ctypes import py_object
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_size_t
from ctypes import c_void_p
from ctypes import memmove
from ctypes.util import find_library
from typing import Union
def putenv(self, name_value, encoding='utf-8'):
    """A wrapper for the pam_putenv function
        Args:
          name_value: environment variable in the format KEY=VALUE
                      Without an '=' delete the corresponding variable
        Returns:
          Linux-PAM return value as int
        """
    if not self.handle:
        return PAM_SYSTEM_ERR
    name_value = name_value.encode(encoding)
    retval = self.pam_putenv(self.handle, name_value)
    if retval != PAM_SUCCESS:
        raise Exception(self.pam_strerror(self.handle, retval))
    return retval