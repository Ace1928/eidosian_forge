from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def has_trigger(self, trigger, state=None):
    """ Check whether an event/trigger is known to the machine
        Args:
            trigger (str): Event/trigger name
            state (optional[NestedState]): Limits the recursive search to this state and its children
        Returns:
            bool: True if event is known and False otherwise
        """
    state = state or self
    return trigger in state.events or any((self.has_trigger(trigger, sta) for sta in state.states.values()))