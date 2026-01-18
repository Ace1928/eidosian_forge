from __future__ import absolute_import
from . import Machines
from .Machines import LOWEST_PRIORITY
from .Transitions import TransitionMap
def new_to_old(self, new_state):
    """Given a new state, return a set of corresponding old states."""
    return self.new_to_old_dict[id(new_state)]