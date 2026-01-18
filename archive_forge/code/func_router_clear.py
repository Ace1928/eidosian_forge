from ctypes import *
from ctypes.util import find_library
import os
def router_clear(self):
    if self.router is not None:
        fluid_midi_router_clear_rules(self.router)