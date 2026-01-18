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
def test_jtimes_native_cvode(nu=0.01, k=1.0, m=1.0, x0=1.0, atol=1e-12, rtol=1e-12):
    w0 = (k / m) ** 0.5

    def f(t, y, p):
        return [y[1], -w0 ** 2 * y[0] - nu * y[1]]
    odesys = NativeSys.from_callback(f, 2, 0, jac=True, jtimes=True)
    tout, yout, info = odesys.integrate(100, [x0, 0], integrator='cvode', with_jtimes=True, with_jacobian=False, method='bdf', linear_solver='gmres', atol=atol, rtol=rtol, nsteps=20000)
    w = (w0 ** 2 - nu ** 2 / 4.0) ** 0.5
    a = (x0 ** 2 + (nu * x0 / 2) ** 2 / w ** 2) ** 0.5
    phi = np.arctan(nu * x0 / (2 * x0 * w))
    ref = a * np.exp(-nu * tout / 2) * np.cos(w * tout - phi)
    assert info['njvev'] > 0
    assert info['njev'] == 0
    assert np.allclose(yout[:, 0], ref)