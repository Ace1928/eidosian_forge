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
@pytest.mark.parametrize('idx1,idx2,scaled,b2', product([0, 1, 2], [0, 1, 2], [True, False], [None, 0]))
def test_TransformedSys__roots(idx1, idx2, scaled, b2):

    def f(x, y, p):
        return [-p[0] * y[0], p[0] * y[0] - p[1] * y[1], p[1] * y[1]]

    def roots(x, y):
        return ([y[0] - 3 * y[1]], [y[0] - 3 * y[2]], [3 * y[1] - y[2]])[idx1]
    if scaled:
        orisys = SymbolicSys.from_callback(f, 3, 2, roots_cb=roots)
    else:
        orisys = ScaledSys.from_callback(f, 3, 2, roots_cb=roots, dep_scaling=42)
    _p, _q, tend = (7, 3, 0.7)
    dep0 = (1, 0.1, 0)
    ref = [0.02969588399749174, 0.1241509730780618, 0.6110670818418275]

    def check(odesys):
        res = odesys.integrate(tend, dep0, (_p, _q), integrator='cvode', return_on_root=True)
        assert abs(res.xout[-1] - ref[idx1]) < 6e-07
    logexp = get_logexp(1, 1e-20, b2=None)
    LogLogSys = symmetricsys(logexp, logexp, check_transforms=False)
    if idx2 == 0:
        check(orisys)
        loglog = LogLogSys.from_other(orisys)
        check(loglog)
        psys1 = PartiallySolvedSystem(orisys, lambda t0, xyz, par0, be: {orisys.dep[0]: xyz[0] * be.exp(-par0[0] * (orisys.indep - t0))})
        check(psys1)
        ploglog1 = LogLogSys.from_other(psys1)
        check(ploglog1)
    psys2 = PartiallySolvedSystem(orisys, lambda t0, iv, p0: {orisys.dep[idx2]: iv[0] + iv[1] + iv[2] - sum((orisys.dep[j] for j in range(3) if j != idx2))})
    ploglog2 = LogLogSys.from_other(psys2)
    check(ploglog2)