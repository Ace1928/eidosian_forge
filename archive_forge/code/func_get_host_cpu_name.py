import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def get_host_cpu_name():
    """
    Get the name of the host's CPU, suitable for using with
    :meth:`Target.create_target_machine()`.
    """
    with ffi.OutputString() as out:
        ffi.lib.LLVMPY_GetHostCPUName(out)
        return str(out)