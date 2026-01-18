from ctypes import *
from ctypes.util import find_library
import os
def bank_select(self, chan, bank):
    """Choose a bank"""
    return fluid_synth_bank_select(self.synth, chan, bank)