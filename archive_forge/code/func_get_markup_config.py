from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def get_markup_config(self):
    """ Generates and returns all machine markup parameters except models.
        Returns:
            dict of machine configuration parameters.
        """
    if self._needs_update:
        self._convert_states_and_transitions(self._markup)
        self._needs_update = False
    return self._markup