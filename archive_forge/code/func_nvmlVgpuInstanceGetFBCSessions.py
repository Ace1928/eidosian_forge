from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetFBCSessions(vgpuInstance):
    c_session_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetFBCSessions')
    ret = fn(vgpuInstance, byref(c_session_count), None)
    if ret == NVML_SUCCESS:
        if c_session_count.value != 0:
            session_array = c_nvmlFBCSession_t * c_session_count.value
            c_sessions = session_array()
            ret = fn(vgpuInstance, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []
    else:
        raise NVMLError(ret)