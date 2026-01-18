from __future__ import (absolute_import, division, print_function)
import os
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ctypes import CDLL, c_char_p, c_int, byref, POINTER, get_errno
def selinux_getpolicytype():
    con = c_char_p()
    try:
        rc = _selinux_lib.selinux_getpolicytype(byref(con))
        return [rc, to_native(con.value)]
    finally:
        _selinux_lib.freecon(con)