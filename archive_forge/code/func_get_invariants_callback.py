from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_invariants_callback(self):
    invar = self.all_invariants()
    if len(invar) == 0:
        return None
    return self._callback_factory(invar)