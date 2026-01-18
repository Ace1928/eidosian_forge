from ctypes import *
from ctypes.util import find_library
import os
def get_chorus_speed(self):
    if fluid_synth_get_chorus_speed is not None:
        return fluid_synth_get_chorus_speed(self.synth)
    else:
        return fluid_synth_get_chorus_speed_Hz(self.synth)