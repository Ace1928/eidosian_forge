from __future__ import (absolute_import, division, print_function)
import numpy as np
from .. import ODESys
from ..util import requires
from .test_core import sine, sine_jac
@requires('pycvodes')
def test_Result_extend_by_integration():
    atol, rtol = (1e-08, 1e-08)
    odesys = ODESys(sine, sine_jac, roots_cb=lambda x, y, p: [y[1]], nroots=1)
    A, k = (2, 3)
    tend = 1
    intkw = dict(integrator='cvode', atol=atol, rtol=rtol)
    result = odesys.integrate(np.linspace(0, tend, 17), [0, A * k], [k], return_on_root=True, **intkw)
    result.extend_by_integration(tend, **intkw)
    assert result.info['success']
    ref = np.array([A * np.sin(k * result.xout), A * np.cos(k * result.xout) * k])
    assert np.allclose(ref.T, result.yout)