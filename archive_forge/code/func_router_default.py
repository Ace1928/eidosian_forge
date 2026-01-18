from ctypes import *
from ctypes.util import find_library
import os
def router_default(self):
    if self.router is not None:
        fluid_midi_router_set_default_rules(self.router)