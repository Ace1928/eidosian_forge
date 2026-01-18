from ctypes import *
from ctypes.util import find_library
import os
def sfload(self, filename, update_midi_preset=0):
    """Load SoundFont and return its ID"""
    return fluid_synth_sfload(self.synth, filename.encode(), update_midi_preset)