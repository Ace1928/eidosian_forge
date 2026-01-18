from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
def exit(self, event_data):
    """ Extends `transitions.core.State.exit` by deleting the temporal object from the model. """
    super(Volatile, self).exit(event_data)
    try:
        delattr(event_data.model, self.volatile_hook)
    except AttributeError:
        pass