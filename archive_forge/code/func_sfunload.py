from ctypes import *
from ctypes.util import find_library
import os
def sfunload(self, sfid, update_midi_preset=0):
    """Unload a SoundFont and free memory it used"""
    return fluid_synth_sfunload(self.synth, sfid, update_midi_preset)