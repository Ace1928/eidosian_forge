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
def test_SymbolicSys_as_autonomous():
    import sympy

    def rhs(t, y, p, backend=math):
        return [y[1], backend.sin(t) - p[0] * y[0]]
    odesys = SymbolicSys.from_callback(rhs, 2, 1)

    def analytic(tout, init_y, p):
        t, (k,) = (odesys.indep, odesys.params)
        c1, c2 = sympy.symbols('c1 c2')
        sqk = sympy.sqrt(k)
        f = c1 * sympy.cos(sqk * t) + c2 * sympy.sin(sqk * t) + sympy.sin(t) / (k - 1)
        dfdt = f.diff(t)
        t0 = tout[0]
        sol, = sympy.solve([f.subs(t, t0) - init_y[0], dfdt.subs(t, t0) - init_y[1]], [c1, c2], dict=True)
        sol[k] = p[0]
        exprs = [f.subs(sol), dfdt.subs(sol)]
        cb = sympy.lambdify([t], exprs)
        return np.array(cb(tout)).T

    def integrate_and_check(system):
        init_y = [0, 0]
        p = [2]
        result = system.integrate([0, 80], init_y, p, integrator='cvode', nsteps=5000)
        yref = analytic(result.xout, init_y, p)
        assert np.all(result.yout - yref < 1.6e-05)
    integrate_and_check(odesys)
    assert len(odesys.dep) == 2
    assert not odesys.autonomous_interface
    assert not odesys.autonomous_exprs
    asys = odesys.as_autonomous()
    integrate_and_check(asys)
    assert len(asys.dep) == 3
    assert not asys.autonomous_interface
    assert asys.autonomous_exprs