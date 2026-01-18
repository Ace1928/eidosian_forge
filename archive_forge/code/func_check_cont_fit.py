import numpy as np
from scipy import stats
from .distparams import distcont
def check_cont_fit(distname, arg):
    distfn = getattr(stats, distname)
    rvs = distfn.rvs(*arg, size=n_repl1)
    est = distfn.fit(rvs)
    truearg = np.hstack([arg, [0.0, 1.0]])
    diff = est - truearg
    txt = ''
    diffthreshold = np.max(np.vstack([truearg * thresh_percent, np.ones(distfn.numargs + 2) * thresh_min]), 0)
    diffthreshold[-2] = np.max([np.abs(rvs.mean()) * thresh_percent, thresh_min])
    if np.any(np.isnan(est)):
        raise AssertionError('nan returned in fit')
    elif np.any(np.abs(diff) - diffthreshold > 0.0):
        rvs = np.concatenate([rvs, distfn.rvs(*arg, size=n_repl2 - n_repl1)])
        est = distfn.fit(rvs)
        truearg = np.hstack([arg, [0.0, 1.0]])
        diff = est - truearg
        if np.any(np.abs(diff) - diffthreshold > 0.0):
            txt = 'parameter: %s\n' % str(truearg)
            txt += 'estimated: %s\n' % str(est)
            txt += 'diff     : %s\n' % str(diff)
            raise AssertionError('fit not very good in %s\n' % distfn.name + txt)