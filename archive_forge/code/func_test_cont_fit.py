import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
@pytest.mark.slow
@pytest.mark.parametrize('distname,arg', cases_test_cont_fit())
@pytest.mark.parametrize('method', ['MLE', 'MM'])
def test_cont_fit(distname, arg, method):
    if distname in failing_fits[method]:
        try:
            xfail = not int(os.environ['SCIPY_XFAIL'])
        except Exception:
            xfail = True
        if xfail:
            msg = "Fitting %s doesn't work reliably yet" % distname
            msg += ' [Set environment variable SCIPY_XFAIL=1 to run this test nevertheless.]'
            pytest.xfail(msg)
    distfn = getattr(stats, distname)
    truearg = np.hstack([arg, [0.0, 1.0]])
    diffthreshold = np.max(np.vstack([truearg * thresh_percent, np.full(distfn.numargs + 2, thresh_min)]), 0)
    for fit_size in fit_sizes:
        np.random.seed(1234)
        with np.errstate(all='ignore'):
            rvs = distfn.rvs(*arg, size=fit_size)
            if method == 'MLE' and distfn.name in mle_use_floc0:
                kwds = {'floc': 0}
            else:
                kwds = {}
            est = distfn.fit(rvs, method=method, **kwds)
            if method == 'MLE':
                data1 = stats.CensoredData(rvs)
                est1 = distfn.fit(data1, **kwds)
                msg = f'Different results fitting uncensored data wrapped as CensoredData: {distfn.name}: est={est} est1={est1}'
                assert_allclose(est1, est, rtol=1e-10, err_msg=msg)
            if method == 'MLE' and distname not in fail_interval_censored:
                nic = 15
                interval = np.column_stack((rvs, rvs))
                interval[:nic, 0] *= 0.99
                interval[:nic, 1] *= 1.01
                interval.sort(axis=1)
                data2 = stats.CensoredData(interval=interval)
                est2 = distfn.fit(data2, **kwds)
                msg = f'Different results fitting interval-censored data: {distfn.name}: est={est} est2={est2}'
                assert_allclose(est2, est, rtol=0.05, err_msg=msg)
        diff = est - truearg
        diffthreshold[-2] = np.max([np.abs(rvs.mean()) * thresh_percent, thresh_min])
        if np.any(np.isnan(est)):
            raise AssertionError('nan returned in fit')
        elif np.all(np.abs(diff) <= diffthreshold):
            break
    else:
        txt = 'parameter: %s\n' % str(truearg)
        txt += 'estimated: %s\n' % str(est)
        txt += 'diff     : %s\n' % str(diff)
        raise AssertionError('fit not very good in %s\n' % distfn.name + txt)