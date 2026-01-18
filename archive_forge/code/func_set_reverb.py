from ctypes import *
from ctypes.util import find_library
import os
def set_reverb(self, roomsize=-1.0, damping=-1.0, width=-1.0, level=-1.0):
    """
        roomsize Reverb room size value (0.0-1.0)
        damping Reverb damping value (0.0-1.0)
        width Reverb width value (0.0-100.0)
        level Reverb level value (0.0-1.0)
        """
    if fluid_synth_set_reverb is not None:
        return fluid_synth_set_reverb(self.synth, roomsize, damping, width, level)
    else:
        set = 0
        if roomsize >= 0:
            set += 1
        if damping >= 0:
            set += 2
        if width >= 0:
            set += 4
        if level >= 0:
            set += 8
        return fluid_synth_set_reverb_full(self.synth, set, roomsize, damping, width, level)