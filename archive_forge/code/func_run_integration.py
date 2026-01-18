from __future__ import division, print_function, absolute_import
import math
import numpy as np
from ..util import import_
from ..core import RecoverableError
from ..symbolic import ScaledSys
def run_integration(inits=(1, 0, 0), rates=(0.04, 10000.0, 30000000.0), t0=1e-10, tend=1e+19, nt=2, logc=False, logt=False, reduced=False, atol=1e-08, rtol=1e-08, zero_conc=1e-23, dep_scaling=1, indep_scaling=1, powsimp=False, wrapping_class=None, **kwargs):
    if nt == 2:
        tout = (t0, tend)
    else:
        tout = np.logspace(np.log10(t0), np.log10(tend), nt)
    names = 'A B C'.split()
    if reduced > 0:
        names.pop(reduced - 1)
    odesys = ScaledSys.from_callback(get_ode_exprs(logc, logt, reduced)[0], 2 if reduced else 3, 6 if reduced else 3, dep_scaling=dep_scaling, indep_scaling=indep_scaling, exprs_process_cb=(lambda exprs: [sp.powsimp(expr.expand(), force=True) for expr in exprs]) if powsimp else None, names=names)
    if wrapping_class is not None:
        finalsys = wrapping_class.from_other(odesys)
    else:
        finalsys = odesys
    indices = {0: (0, 1, 2), 1: (1, 2), 2: (0, 2), 3: (0, 1)}[reduced]
    _inits = np.array([inits[idx] for idx in indices], dtype=np.float64)
    if logc:
        _inits = np.log(_inits + zero_conc)
    if logt:
        tout = np.log(tout)
    xout, yout, info = odesys.integrate(tout, _inits, rates + inits if reduced else rates, atol=atol, rtol=rtol, **kwargs)
    if logc:
        yout = np.exp(yout)
    if logt:
        xout = np.exp(xout)
    if reduced:
        yout = np.insert(yout, reduced - 1, 1 - np.sum(yout, axis=1), axis=1)
    return (xout, yout, info, (odesys, finalsys))