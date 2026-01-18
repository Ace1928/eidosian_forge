from ctypes import *
from ctypes.util import find_library
import os
def set_reverb_roomsize(self, roomsize):
    if fluid_synth_set_reverb_roomsize is not None:
        return fluid_synth_set_reverb_roomsize(self.synth, roomsize)
    else:
        return self.set_reverb(roomsize=roomsize)