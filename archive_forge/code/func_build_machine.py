from __future__ import absolute_import
import types
from . import Errors
def build_machine(self, m, initial_state, final_state, match_bol, nocase):
    self.re.build_machine(m, initial_state, final_state, match_bol, self.nocase)