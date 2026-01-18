from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def get_state(self, state, hint=None):
    """ Return the State instance with the passed name.
        Args:
            state (str, Enum or list(str)): A state name, enum or state path
            hint (list(str)): A state path to check for the state in question
        Returns:
            NestedState that belongs to the passed str (list) or Enum.
        """
    if isinstance(state, Enum):
        state = self._get_enum_path(state)
    elif isinstance(state, string_types):
        state = state.split(self.state_cls.separator)
    if not hint:
        state = copy.copy(state)
        hint = copy.copy(state)
    if len(state) > 1:
        child = state.pop(0)
        try:
            with self(child):
                return self.get_state(state, hint)
        except (KeyError, ValueError):
            try:
                with self():
                    state = self
                    for elem in hint:
                        state = state.states[elem]
                    return state
            except KeyError:
                raise ValueError("State '%s' is not a registered state." % self.state_cls.separator.join(hint))
    elif state[0] not in self.states:
        raise ValueError("State '%s' is not a registered state." % state)
    return self.states[state[0]]