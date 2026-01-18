import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
@pytest.mark.parametrize('distname,arg,first_case', cases_test_discrete_basic())
def test_discrete_basic(distname, arg, first_case):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'sample distribution'
    np.random.seed(9765456)
    rvs = distfn.rvs(*arg, size=2000)
    supp = np.unique(rvs)
    m, v = distfn.stats(*arg)
    check_cdf_ppf(distfn, arg, supp, distname + ' cdf_ppf')
    check_pmf_cdf(distfn, arg, distname)
    check_oth(distfn, arg, supp, distname + ' oth')
    check_edge_support(distfn, arg)
    alpha = 0.01
    check_discrete_chisquare(distfn, arg, rvs, alpha, distname + ' chisquare')
    if first_case:
        locscale_defaults = (0,)
        meths = [distfn.pmf, distfn.logpmf, distfn.cdf, distfn.logcdf, distfn.logsf]
        spec_k = {'randint': 11, 'hypergeom': 4, 'bernoulli': 0, 'nchypergeom_wallenius': 6}
        k = spec_k.get(distname, 1)
        check_named_args(distfn, k, arg, locscale_defaults, meths)
        if distname != 'sample distribution':
            check_scale_docstring(distfn)
        check_random_state_property(distfn, arg)
        check_pickling(distfn, arg)
        check_freezing(distfn, arg)
        check_entropy(distfn, arg, distname)
        if distfn.__class__._entropy != stats.rv_discrete._entropy:
            check_private_entropy(distfn, arg, stats.rv_discrete)