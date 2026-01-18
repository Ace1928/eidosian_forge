from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
def calc_length(self, length):
    return length % 2