from ctypes import *
from ctypes.util import find_library
import os
def program_info(self, chan):
    """get active soundfont, bank, prog on a channel"""
    if fluid_synth_get_program is not None:
        sfontid = c_int()
        banknum = c_int()
        presetnum = c_int()
        fluid_synth_get_program(self.synth, chan, byref(sfontid), byref(banknum), byref(presetnum))
        return (sfontid.value, banknum.value, presetnum.value)
    else:
        sfontid, banknum, prognum, presetname = self.channel_info(chan)
        return (sfontid, banknum, prognum)