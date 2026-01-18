from ctypes import *
from ctypes.util import find_library
import os
def router_begin(self, type):
    """types are [note|cc|prog|pbend|cpress|kpress]"""
    if self.router is not None:
        if type == 'note':
            self.router.cmd_rule_type = 0
        elif type == 'cc':
            self.router.cmd_rule_type = 1
        elif type == 'prog':
            self.router.cmd_rule_type = 2
        elif type == 'pbend':
            self.router.cmd_rule_type = 3
        elif type == 'cpress':
            self.router.cmd_rule_type = 4
        elif type == 'kpress':
            self.router.cmd_rule_type = 5
        if 'self.router.cmd_rule' in globals():
            delete_fluid_midi_router_rule(self.router.cmd_rule)
        self.router.cmd_rule = new_fluid_midi_router_rule()