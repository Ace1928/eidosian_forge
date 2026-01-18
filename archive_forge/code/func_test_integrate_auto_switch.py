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
@pytest.mark.parametrize('integrator,method', [('cvode', 'adams'), ('gsl', 'msadams')])
def test_integrate_auto_switch(integrator, method):
    for p in (0, 1, 2, 3):
        n, a = (7, 5)
        atol, rtol = (1e-10, 1e-10)
        y0, k, linsys = get_special_chain(n, p, a)
        y0 += 1e-10
        LogLogSys = symmetricsys(get_logexp(), get_logexp())
        logsys = LogLogSys.from_other(linsys)
        tout = [10 ** (-12), 1]
        kw = dict(integrator=integrator, method=method, atol=atol, rtol=rtol)
        forgive = (5 + p) * 1.2
        xout, yout, info = integrate_auto_switch([logsys, linsys], {'nsteps': [1, 1]}, tout, y0, return_on_error=True, **kw)
        assert info['success'] == False
        ntot = 400
        nlinear = 60 * (p + 3)
        xout, yout, info = integrate_auto_switch([logsys, linsys], {'nsteps': [ntot - nlinear, nlinear], 'first_step': [30.0, 1e-05], 'return_on_error': [True, False]}, tout, y0, **kw)
        assert info['success'] == True
        check(yout[-1, :], n, p, a, atol, rtol, forgive)