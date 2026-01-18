from __future__ import (absolute_import, division, print_function)
import numpy as np
from ..util import import_
import pytest
from .. import ODESys
from ..core import integrate_chained
from ..symbolic import SymbolicSys, PartiallySolvedSystem, symmetricsys
from ..util import requires, pycvodes_double
from ._robertson import run_integration, get_ode_exprs
@requires('sym', 'sympy', 'pycvodes')
@pycvodes_double
def test_integrate_chained_multi_robertson():
    odes = logsys, linsys = [ODESys(*get_ode_exprs(l, l)) for l in [True, False]]

    def pre(x, y, p):
        return (np.log(x), np.log(y), p)

    def post(x, y, p):
        return (np.exp(x), np.exp(y), p)
    logsys.pre_processors = [pre]
    logsys.post_processors = [post]
    zero_time, zero_conc = (1e-10, 1e-18)
    init_conc = (1, zero_conc, zero_conc)
    k = (0.04, 10000.0, 30000000.0)
    for sys_iter, kw in [(odes, {'nsteps': [100, 1660], 'return_on_error': [True, False]}), (odes[1:], {'nsteps': [1705 * 1.01]})]:
        results = integrate_chained(sys_iter, kw, [(zero_time, 100000000000.0)] * 3, [init_conc] * 3, [k + init_conc] * 3, integrator='cvode', atol=1e-10, rtol=1e-14, first_step=1e-14)
        assert len(results) == 3
        for res in results:
            x, y, nfo = res
            assert np.allclose(_yref_1e11, y[-1, :], atol=1e-16, rtol=0.02)
            assert nfo['success'] is True
            assert nfo['nfev'] > 100
            assert nfo['njev'] > 10
            assert nfo['nsys'] in (1, 2)