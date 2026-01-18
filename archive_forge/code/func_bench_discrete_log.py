import sys
from time import time
from sympy.ntheory.residue_ntheory import (discrete_log,
def bench_discrete_log(data_set, algo=None):
    if algo is None:
        f = discrete_log
    elif algo == 'trial':
        f = _discrete_log_trial_mul
    elif algo == 'shanks':
        f = _discrete_log_shanks_steps
    elif algo == 'rho':
        f = _discrete_log_pollard_rho
    elif algo == 'ph':
        f = _discrete_log_pohlig_hellman
    else:
        raise ValueError("Argument 'algo' should be one of ('trial', 'shanks', 'rho' or 'ph')")
    for i, data in enumerate(data_set):
        for j, (n, p, g) in enumerate(data):
            t = time()
            l = f(n, pow(g, p - 1, n), g, p)
            t = time() - t
            print('[%02d-%03d] %15.10f' % (i, j, t))
            assert l == p - 1