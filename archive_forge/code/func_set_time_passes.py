from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def set_time_passes(enable):
    """Enable or disable the pass timers.

    Parameters
    ----------
    enable : bool
        Set to True to enable the pass timers.
        Set to False to disable the pass timers.
    """
    ffi.lib.LLVMPY_SetTimePasses(c_bool(enable))