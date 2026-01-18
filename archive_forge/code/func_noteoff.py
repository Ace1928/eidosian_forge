from ctypes import *
from ctypes.util import find_library
import os
def noteoff(self, chan, key):
    """Stop a note"""
    if key < 0 or key > 127:
        return False
    if chan < 0:
        return False
    return fluid_synth_noteoff(self.synth, chan, key)