from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def scoped_exit(self, event_data, scope=None):
    """ Exits a state with the provided scope.
        Args:
            event_data (NestedEventData): The currently processed event.
            scope (list(str)): Names of the state's parents starting with the top most parent.
        """
    self._scope = scope or []
    try:
        self.exit(event_data)
    finally:
        self._scope = []