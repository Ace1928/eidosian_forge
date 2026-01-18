from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@property
def issock(self):
    return self.filetype & 61440 == 49152