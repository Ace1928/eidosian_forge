from ctypes import *
from ctypes.util import find_library
import os
def all_notes_off(self, chan):
    """Turn off all notes on a channel (release all keys)"""
    return fluid_synth_all_notes_off(self.synth, chan)