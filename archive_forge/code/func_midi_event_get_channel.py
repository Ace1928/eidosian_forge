from ctypes import *
from ctypes.util import find_library
import os
def midi_event_get_channel(self, event):
    return fluid_midi_event_get_channel(event)