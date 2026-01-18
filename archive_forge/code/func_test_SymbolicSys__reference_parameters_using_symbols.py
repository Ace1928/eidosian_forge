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
@requires('sym', 'pyodeint')
@pytest.mark.parametrize('method', ['bs', 'rosenbrock4'])
def test_SymbolicSys__reference_parameters_using_symbols(method):
    be = sym.Backend('sympy')
    x, p = map(be.Symbol, 'x p'.split())
    symsys = SymbolicSys([(x, -p * x)], params=True)
    tout = [0, 1e-09, 1e-07, 1e-05, 0.001, 0.1]
    for y_symb in [False, True]:
        for p_symb in [False, True]:
            xout, yout, info = symsys.integrate(tout, {x: 2} if y_symb else [2], {p: 3} if p_symb else [3], method=method, integrator='odeint', atol=1e-12, rtol=1e-12)
            assert np.allclose(yout[:, 0], 2 * np.exp(-3 * xout))