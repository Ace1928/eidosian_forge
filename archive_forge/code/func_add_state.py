import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def add_state(self, states, on_enter=None, on_exit=None, ignore_invalid_triggers=None, **kwargs):
    """ Alias for add_states. """
    self.add_states(states=states, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)