from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
@property
def markup(self):
    """ Returns the machine's configuration as a markup dictionary.
        Returns:
            dict of machine configuration parameters.
        """
    self._markup['models'] = self._convert_models()
    return self.get_markup_config()