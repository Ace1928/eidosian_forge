from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_jtimes_callback(self):
    """ Generate a callback fro evaluating the jacobian-vector product."""
    jtimes = self.get_jtimes()
    if jtimes is False:
        return None
    v, jtimes_exprs = jtimes
    return _Callback(self.indep, tuple(self.dep) + tuple(v), self.params, jtimes_exprs, Lambdify=self.be.Lambdify)