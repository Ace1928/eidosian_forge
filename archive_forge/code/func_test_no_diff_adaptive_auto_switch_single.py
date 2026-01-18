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
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_no_diff_adaptive_auto_switch_single(integrator):
    odesys = _get_decay3()
    tout, y0, k = ([3, 5], [3, 2, 1], [3.5, 2.5, 1.5])
    xout1, yout1, info1 = odesys.integrate(tout, y0, k, integrator=integrator)
    ref = np.array(bateman_full(y0, k, xout1 - xout1[0], exp=np.exp)).T
    assert info1['success']
    assert xout1.size > 10
    assert xout1.size == yout1.shape[0]
    assert np.allclose(yout1, ref)
    xout2, yout2, info2 = integrate_auto_switch([odesys], {}, tout, y0, k, integrator=integrator)
    assert info1['success']
    assert xout2.size == xout1.size
    assert np.allclose(yout2, ref)