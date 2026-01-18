from ctypes import *
from ctypes.util import find_library
import os
def play_midi_stop(self):
    status = fluid_player_stop(self.player)
    if status == FLUID_FAILED:
        return status
    status = fluid_player_seek(self.player, 0)
    delete_fluid_player(self.player)
    return status