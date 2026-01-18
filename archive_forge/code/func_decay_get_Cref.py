from collections import defaultdict
from functools import reduce
from operator import mul
import pytest
from chempy import ReactionSystem
from chempy.units import (
from chempy.util.testing import requires
from ..integrated import binary_rev
from ..ode import get_odesys
from .._native import get_native
def decay_get_Cref(k, y0, tout):
    coeffs = list(k) + [0] * (3 - len(k))
    return np.column_stack([decay_analytic[i](y0, coeffs, tout) for i in range(min(3, len(k) + 1))])