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
@requires('sym', 'scipy')
def test_ScaledSys_from_callback():

    def f(t, x, k):
        return [-k[0] * x[0], k[0] * x[0] - k[1] * x[1], k[1] * x[1] - k[2] * x[2], k[2] * x[2]]
    odesys = ScaledSys.from_callback(f, 4, 3, 314000000.0)
    k = [7, 3, 2]
    y0 = [0] * (len(k) + 1)
    y0[0] = 1
    xout, yout, info = odesys.integrate([1e-12, 1], y0, k, integrator='scipy')
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=3e-11, atol=3e-11)
    with pytest.raises(TypeError):
        odesys.integrate([1e-12, 1], [0] * len(k), k, integrator='scipy')