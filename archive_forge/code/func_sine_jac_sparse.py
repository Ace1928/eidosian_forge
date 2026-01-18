from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
def sine_jac_sparse(t, y, p):
    from scipy.sparse import csc_matrix
    k = p[0]
    Jmat = np.zeros((2, 2))
    Jmat[0, 0] = 0
    Jmat[0, 1] = 1
    Jmat[1, 0] = -k ** 2
    Jmat[1, 1] = 0
    return csc_matrix(Jmat)