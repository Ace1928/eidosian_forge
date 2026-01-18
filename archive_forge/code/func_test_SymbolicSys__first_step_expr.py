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
@requires('sym', 'sympy', 'pycvodes')
def test_SymbolicSys__first_step_expr():
    import sympy
    tend, k, y0 = (5, [1e+23, 3], (0.7, 0.0, 0.0))
    kwargs = dict(integrator='cvode', atol=1e-08, rtol=1e-08)
    factory = decay_dydt_factory(k)
    dep = sympy.symbols('y0 y1 y2', real=True)
    exprs = factory(k, dep)
    odesys = SymbolicSys(zip(dep, exprs), jac=True, first_step_expr=dep[0] * 1e-30)
    xout, yout, info = odesys.integrate(tend, y0, **kwargs)
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, atol=10 * kwargs['atol'], rtol=10 * kwargs['rtol'])