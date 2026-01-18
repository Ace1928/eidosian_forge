from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@linear_invariants.setter
def linear_invariants(self, lin_invar):
    self._linear_invariants = _get_lin_invar_mtx(lin_invar, self.be, self.ny, self.names)