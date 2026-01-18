from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_dfdx(self):
    """ Calculates 2nd derivatives of ``self.exprs`` """
    if self._dfdx is True:
        if self.indep is None:
            zero = 0 * self.be.Dummy() ** 0
            self._dfdx = self.be.Matrix(1, self.ny, [zero] * self.ny)
        else:
            self._dfdx = self.be.Matrix(1, self.ny, [expr.diff(self.indep) for expr in self.exprs])
    elif self._dfdx is False:
        return False
    return self._dfdx