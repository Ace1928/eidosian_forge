from __future__ import absolute_import
from . import Machines
from .Machines import LOWEST_PRIORITY
from .Transitions import TransitionMap
def epsilon_closure(state):
    """
    Return the set of states reachable from the given state
    by epsilon moves.
    """
    result = state.epsilon_closure
    if result is None:
        result = {}
        state.epsilon_closure = result
        add_to_epsilon_closure(result, state)
    return result