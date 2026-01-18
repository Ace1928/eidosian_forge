from __future__ import print_function, absolute_import, division
from collections import defaultdict, OrderedDict
from itertools import product
import math
import numpy as np
import pytest
from .. import ODESys
from ..core import integrate_auto_switch, chained_parameter_variation
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires, pycvodes_double, pycvodes_klu
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa
@requires('sym', 'pycvodes')
def test_ScaledSys():
    import sympy as sp
    k = k0, k1, k2 = [7.0, 3, 2]
    y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3', real=True, positive=True)
    l = [(y0, -7 * y0), (y1, 7 * y0 - 3 * y1), (y2, 3 * y1 - 2 * y2), (y3, 2 * y2)]
    ss = ScaledSys(l, dep_scaling=100000000.0)
    y0 = [0] * (len(k) + 1)
    y0[0] = 1
    xout, yout, info = ss.integrate([1e-12, 1], y0, integrator='cvode', atol=1e-12, rtol=1e-12, nsteps=1000)
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=2e-11, atol=2e-11)