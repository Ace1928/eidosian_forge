from ctypes import *
from ctypes.util import find_library
import os
def midi_event_get_velocity(self, event):
    return fluid_midi_event_get_velocity(event)