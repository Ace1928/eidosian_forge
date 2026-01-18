from ctypes import *
from ctypes.util import find_library
import os
def get_chorus_depth(self):
    if fluid_synth_get_chorus_depth is not None:
        return fluid_synth_get_chorus_depth(self.synth)
    else:
        return fluid_synth_get_chorus_depth_ms(self.synth)