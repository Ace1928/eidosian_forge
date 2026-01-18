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
@pytest.mark.parametrize('idx', [0, 1, 2])
def test_PartiallySolvedSystem__roots(idx):
    import sympy as sp
    t, x, y, z, p, q = sp.symbols('t x y z, p, q')
    odesys = SymbolicSys({x: -p * x, y: p * x - q * y, z: q * y}, t, params=(p, q), roots=([x - y], [x - z], [y - z])[idx])
    _p, _q, tend = (7, 3, 0.7)
    dep0 = {x: 1, y: 0, z: 0}
    ref = [0.11299628093544488, 0.20674119231833346, 0.3541828705348678]

    def check(odesys):
        res = odesys.integrate(tend, [dep0[k] for k in getattr(odesys, 'original_dep', odesys.dep)], (_p, _q), integrator='cvode', return_on_root=True)
        assert abs(res.xout[-1] - ref[idx]) < 1e-07
    check(odesys)
    psys = PartiallySolvedSystem(odesys, lambda t0, xyz, par0: {x: xyz[odesys.dep.index(x)] * sp.exp(-p * (t - t0))})
    check(psys)