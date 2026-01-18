from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def set_state(self, state, model=None):
    """ Set the current state.
        Args:
            state (list of str or Enum or State): value of state(s) to be set
            model (optional[object]): targeted model; if not set, all models will be set to 'state'
        """
    values = [self._set_state(value) for value in listify(state)]
    models = self.models if model is None else listify(model)
    for mod in models:
        setattr(mod, self.model_attribute, values if len(values) > 1 else values[0])