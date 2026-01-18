from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_jtimes(self):
    """ Derive the jacobian-vector product from ``self.exprs`` and ``self.dep``"""
    if self._jtimes is False:
        return False
    if self._jtimes is True:
        r = self.be.Dummy('r')
        v = tuple((self.be.Dummy('v_{0}'.format(i)) for i in range(self.ny)))
        f = self.be.Matrix(1, self.ny, self.exprs)
        f = f.subs([(x_i, x_i + r * v_i) for x_i, v_i in zip(self.dep, v)])
        return (v, self.be.flatten(f.diff(r).subs(r, 0)))
    else:
        return tuple(zip(*self._jtimes))