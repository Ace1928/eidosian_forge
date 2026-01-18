from __future__ import absolute_import
from . import Machines
from .Machines import LOWEST_PRIORITY
from .Transitions import TransitionMap
def highest_priority_action(self, state_set):
    best_action = None
    best_priority = LOWEST_PRIORITY
    for state in state_set:
        priority = state.action_priority
        if priority > best_priority:
            best_action = state.action
            best_priority = priority
    return best_action