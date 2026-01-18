import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def set_defaulted_states(self):
    self.defaulted_states = {}
    for state, actions in self.action.items():
        rules = list(actions.values())
        if len(rules) == 1 and rules[0] < 0:
            self.defaulted_states[state] = rules[0]