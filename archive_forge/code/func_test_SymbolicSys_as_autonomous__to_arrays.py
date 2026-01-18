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
@requires('sym', 'pycvodes', 'quantities')
@pytest.mark.parametrize('auto', [False, True])
def test_SymbolicSys_as_autonomous__to_arrays(auto):
    import quantities as pq

    def rhs(t, y, p):
        k = t ** p[0]
        return [-k * y[0], k * y[0]]

    def analytic(tout, init_y, params):
        y0ref = init_y[0] * np.exp(-tout ** (params[0] + 1) / (params[0] + 1))
        return np.array([y0ref, init_y[0] - y0ref + init_y[1]]).T
    odes = SymbolicSys.from_callback(rhs, 2, 1, to_arrays_callbacks=(lambda t: [_t.rescale(pq.s).magnitude for _t in t], lambda y: [_y.rescale(pq.molar).magnitude for _y in y], None))
    odesys = odes.as_autonomous() if auto else odes
    for from_other in [False, True]:
        if from_other:
            odesys = SymbolicSys.from_other(odesys)
        result = odesys.integrate(4 * pq.s, [5 * pq.molar, 2 * pq.molar], [3], integrator='cvode')
        ref = analytic(result.xout, result.yout[0, :], result.params)
        assert np.allclose(result.yout, ref, atol=1e-06)