import numpy as np
from scipy import stats, optimize, special
def hess_ndt(fun, pars, args, options):
    import numdifftools as ndt
    if not ('stepMax' in options or 'stepFix' in options):
        options['stepMax'] = 1e-05
    f = lambda params: fun(params, *args)
    h = ndt.Hessian(f, **options)
    return (h(pars), h)