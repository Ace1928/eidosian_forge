from ctypes import *
from ctypes.util import find_library
import os
def get_chorus_nr(self):
    return fluid_synth_get_chorus_nr(self.synth)