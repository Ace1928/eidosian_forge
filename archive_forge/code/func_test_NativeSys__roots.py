from __future__ import (absolute_import, division, print_function)
import numpy as np
import pytest
from pyodesys.util import requires, pycvodes_double, pycvodes_klu
from pyodesys.symbolic import SymbolicSys, PartiallySolvedSystem
from ._tests import (
from ._test_robertson_native import _test_chained_multi_native
from ..cvode import NativeCvodeSys as NativeSys
from pyodesys.tests.test_symbolic import _test_chained_parameter_variation
@requires('pycvodes')
def test_NativeSys__roots():

    def f(t, y):
        return [y[0]]

    def roots(t, y, p, backend):
        return [y[0] - backend.exp(1)]
    odesys = NativeSys.from_callback(f, 1, roots_cb=roots)
    kwargs = dict(first_step=1e-12, atol=1e-12, rtol=1e-12, method='adams', integrator='cvode')
    for return_on_root in (False, True):
        result = odesys.integrate(2, [1], **kwargs)
        assert len(result.info['root_indices']) == 1
        assert result.info['success'] == True
        assert np.min(np.abs(result.xout - 1)) < 1e-11