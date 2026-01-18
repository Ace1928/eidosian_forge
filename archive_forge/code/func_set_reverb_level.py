from ctypes import *
from ctypes.util import find_library
import os
def set_reverb_level(self, level):
    if fluid_synth_set_reverb_level is not None:
        return fluid_synth_set_reverb_level(self.synth, level)
    else:
        return self.set_reverb(level=level)