from ctypes import *
from ctypes.util import find_library
import os
def channel_info(self, chan):
    """get soundfont, bank, prog, preset name of channel"""
    if fluid_synth_get_channel_info is not None:
        info = fluid_synth_channel_info_t()
        fluid_synth_get_channel_info(self.synth, chan, byref(info))
        return (info.sfont_id, info.bank, info.program, info.name)
    else:
        sfontid, banknum, presetnum = self.program_info(chan)
        presetname = self.sfpreset_name(sfontid, banknum, presetnum)
        return (sfontid, banknum, presetnum, presetname)