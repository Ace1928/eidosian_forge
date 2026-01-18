from ctypes import *
from ctypes.util import find_library
import os
def set_chorus_speed(self, speed):
    if fluid_synth_set_chorus_speed is not None:
        return fluid_synth_set_chorus_speed(self.synth, speed)
    else:
        return self.set_chorus(speed=speed)