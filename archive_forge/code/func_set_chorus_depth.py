from ctypes import *
from ctypes.util import find_library
import os
def set_chorus_depth(self, depth):
    if fluid_synth_set_chorus_depth is not None:
        return fluid_synth_set_chorus_depth(self.synth, depth)
    else:
        return self.set_chorus(depth=depth)