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
@pytest.mark.parametrize('reduced_nsteps', [(0, [(1, 1705 * 1.01), (4988 * 1.01, 1), (200, 1633), (4988 * 0.69, 1705 * 0.69)]), (1, [(1, 1563 * 1.1), (100, 1700 * 1.01)]), (2, [(1, 1674 * 1.1), (100, 1597 * 1.1)]), (3, [(1, 1591 * 1.1), (5000, 1), (100, 1600), (4572 * 0.66, 1100)])])
def test_integrate_chained_robertson(reduced_nsteps):
    reduced, all_nsteps = reduced_nsteps
    rtols = {0: 0.02, 1: 0.1, 2: 0.02, 3: 0.015}
    odes = logsys, linsys = [ODESys(*get_ode_exprs(l, l, reduced=reduced)) for l in [True, False]]

    def pre(x, y, p):
        return (np.log(x), np.log(y), p)

    def post(x, y, p):
        return (np.exp(x), np.exp(y), p)
    logsys.pre_processors = [pre]
    logsys.post_processors = [post]
    zero_time, zero_conc = (1e-10, 1e-18)
    init_conc = (1, zero_conc, zero_conc)
    k = (0.04, 10000.0, 30000000.0)
    for nsteps in all_nsteps:
        y0 = [_ for i, _ in enumerate(init_conc) if i != reduced - 1]
        _atol = [1e-10] * 3
        x, y, nfo = integrate_chained(odes, {'nsteps': nsteps, 'return_on_error': [True, False]}, (zero_time, 100000000000.0), y0, k + init_conc, integrator='cvode', atol=[at for i, at in enumerate(_atol) if i != reduced - 1], rtol=1e-14, first_step=1e-12)
        if reduced > 0:
            y = np.insert(y, reduced - 1, init_conc[0] - np.sum(y, axis=1), axis=1)
        assert np.allclose(_yref_1e11, y[-1, :], atol=_atol, rtol=rtols[reduced])
        assert nfo['success'] == True
        assert nfo['nfev'] > 100
        assert nfo['njev'] > 10
    with pytest.raises(KeyError):
        nfo['asdjklda']