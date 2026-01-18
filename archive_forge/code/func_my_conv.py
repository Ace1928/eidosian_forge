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
def my_conv(n_messages, messages, p_response, libc, msg_list: list, password: bytes, encoding: str):
    """Simple conversation function that responds to any
       prompt where the echo is off with the supplied password"""
    calloc = libc.calloc
    calloc.restype = c_void_p
    calloc.argtypes = [c_size_t, c_size_t]
    cpassword = c_char_p(password)
    '\n    PAM_PROMPT_ECHO_OFF = 1\n    PAM_PROMPT_ECHO_ON = 2\n    PAM_ERROR_MSG = 3\n    PAM_TEXT_INFO = 4\n    '
    addr = calloc(n_messages, sizeof(PamResponse))
    response = cast(addr, POINTER(PamResponse))
    p_response[0] = response
    for i in range(n_messages):
        message = messages[i].contents.msg
        if sys.version_info >= (3,):
            message = message.decode(encoding)
        msg_list.append(message)
        if messages[i].contents.msg_style == PAM_PROMPT_ECHO_OFF:
            if i == 0:
                dst = calloc(len(password) + 1, sizeof(c_char))
                memmove(dst, cpassword, len(password))
                response[i].resp = dst
            else:
                response[i].resp = None
            response[i].resp_retcode = 0
    return PAM_SUCCESS