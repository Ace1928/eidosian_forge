from __future__ import (absolute_import, division, print_function)
import numpy as np
import pytest
from pyodesys.util import requires, pycvodes_double, pycvodes_klu
from pyodesys.symbolic import SymbolicSys, PartiallySolvedSystem
from ._tests import (
from ._test_robertson_native import _test_chained_multi_native
from ..cvode import NativeCvodeSys as NativeSys
from pyodesys.tests.test_symbolic import _test_chained_parameter_variation
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('idx', [0, 1, 2])
def test_NativeSys__PartiallySolvedSystem__roots(idx):

    def f(x, y, p):
        return [-p[0] * y[0], p[0] * y[0] - p[1] * y[1], p[1] * y[1]]

    def roots(x, y):
        return ([y[0] - y[1]], [y[0] - y[2]], [y[1] - y[2]])[idx]
    odesys = SymbolicSys.from_callback(f, 3, 2, roots_cb=roots)
    _p, _q, tend = (7, 3, 0.7)
    dep0 = (1, 0, 0)
    ref = [0.11299628093544488, 0.20674119231833346, 0.3541828705348678]

    def check(odesys):
        res = odesys.integrate(tend, dep0, (_p, _q), integrator='cvode', return_on_root=True)
        assert abs(res.xout[-1] - ref[idx]) < 1e-07
    check(odesys)
    native = NativeSys.from_other(odesys)
    check(native)
    psys = PartiallySolvedSystem(odesys, lambda t0, xyz, par0, be: {odesys.dep[0]: xyz[0] * be.exp(-par0[0] * (odesys.indep - t0))})
    check(psys)
    pnative = NativeSys.from_other(psys)
    check(pnative)