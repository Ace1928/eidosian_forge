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
@pytest.mark.parametrize('n', [29])
def test_long_chain_banded_scipy(n):
    p, a = (0, n)
    y0, k, odesys_dens = get_special_chain(n, p, a)
    y0, k, odesys_band = get_special_chain(n, p, a, band=(1, 0))
    atol, rtol = (1e-07, 1e-07)
    tout = np.logspace(-10, 0, 10)

    def mk_callback(odesys):

        def callback(*args, **kwargs):
            return odesys.integrate(*args, integrator='scipy', **kwargs)
        return callback
    min_time_dens, min_time_band = (float('inf'), float('inf'))
    for _ in range(3):
        time_dens, (xout_dens, yout_dens, info) = timeit(mk_callback(odesys_dens), tout, y0, atol=atol, rtol=rtol, name='vode', method='bdf', first_step=1e-10)
        assert info['njev'] > 0
        min_time_dens = min(min_time_dens, time_dens)
    for _ in range(3):
        time_band, (xout_band, yout_band, info) = timeit(mk_callback(odesys_band), tout, y0, atol=atol, rtol=rtol, name='vode', method='bdf', first_step=1e-10)
        assert info['njev'] > 0
        min_time_band = min(min_time_band, time_band)
    check(yout_dens[-1, :], n, p, a, atol, rtol, 1.5)
    check(yout_band[-1, :], n, p, a, atol, rtol, 1.5)
    assert min_time_dens * 2 > min_time_band