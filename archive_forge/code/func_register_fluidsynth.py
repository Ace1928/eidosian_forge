from ctypes import *
from ctypes.util import find_library
import os
def register_fluidsynth(self, synth):
    response = fluid_sequencer_register_fluidsynth(self.sequencer, synth.synth)
    if response == FLUID_FAILED:
        raise Error('Registering fluid synth failed')
    return response