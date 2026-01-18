from ctypes import *
from ctypes.util import find_library
import os
def sfpreset_name(self, sfid, bank, prenum):
    """Return name of a soundfont preset"""
    if fluid_synth_get_sfont_by_id is not None:
        sfont = fluid_synth_get_sfont_by_id(self.synth, sfid)
        preset = fluid_sfont_get_preset(sfont, bank, prenum)
        if not preset:
            return None
        return fluid_preset_get_name(preset).decode('ascii')
    else:
        sfontid, banknum, presetnum, presetname = self.channel_info(chan)
        return presetname