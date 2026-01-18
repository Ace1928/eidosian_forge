from ctypes import *
from ctypes.util import find_library
import os
def get_tick(self):
    return fluid_sequencer_get_tick(self.sequencer)