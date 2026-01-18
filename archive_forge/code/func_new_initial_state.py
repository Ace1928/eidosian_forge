from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
def new_initial_state(self, name):
    state = self.new_state()
    self.make_initial_state(name, state)
    return state