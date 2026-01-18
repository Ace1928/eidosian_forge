from __future__ import absolute_import
from . import Machines
from .Machines import LOWEST_PRIORITY
from .Transitions import TransitionMap
def set_epsilon_closure(state_set):
    """
    Given a set of states, return the union of the epsilon
    closures of its member states.
    """
    result = {}
    for state1 in state_set:
        for state2 in epsilon_closure(state1):
            result[state2] = 1
    return result