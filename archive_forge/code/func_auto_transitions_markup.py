from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
@auto_transitions_markup.setter
def auto_transitions_markup(self, value):
    """ Whether auto transitions should be included in the markup. """
    self._auto_transitions_markup = value
    self._needs_update = True