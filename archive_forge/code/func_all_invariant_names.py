from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def all_invariant_names(self):
    return (self.linear_invariant_names or []) + (self.nonlinear_invariant_names or [])