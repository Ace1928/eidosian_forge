from ctypes import *
from ctypes.util import find_library
import os
def set_reverb_width(self, width):
    if fluid_synth_set_reverb_width is not None:
        return fluid_synth_set_reverb_width(self.synth, width)
    else:
        return self.set_reverb(width=width)