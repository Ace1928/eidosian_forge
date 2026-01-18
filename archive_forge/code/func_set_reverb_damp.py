from ctypes import *
from ctypes.util import find_library
import os
def set_reverb_damp(self, damping):
    if fluid_synth_set_reverb_damp is not None:
        return fluid_synth_set_reverb_damp(self.synth, damping)
    else:
        return self.set_reverb(damping=damping)