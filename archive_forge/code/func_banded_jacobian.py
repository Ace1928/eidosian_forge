from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
def banded_jacobian(self, exprs, dep, ml, mu):
    """ Wraps Matrix around result of .util.banded_jacobian """
    exprs = banded_jacobian(exprs, dep, ml, mu)
    return self.Matrix(ml + mu + 1, len(dep), list(exprs.flat))