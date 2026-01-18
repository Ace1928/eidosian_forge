from ctypes import *
from ctypes.util import find_library
import os
def noteon(self, chan, key, vel):
    """Play a note"""
    if key < 0 or key > 127:
        return False
    if chan < 0:
        return False
    if vel < 0 or vel > 127:
        return False
    return fluid_synth_noteon(self.synth, chan, key, vel)