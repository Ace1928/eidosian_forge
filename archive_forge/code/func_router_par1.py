from ctypes import *
from ctypes.util import find_library
import os
def router_par1(self, min, max, mul, add):
    if self.router is not None:
        fluid_midi_router_rule_set_param1(self.router.cmd_rule, min, max, mul, add)