from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_roots_callback(self):
    """ Generate a callback for evaluating ``self.roots`` """
    if self.roots is None:
        return None
    return self._callback_factory(self.roots)