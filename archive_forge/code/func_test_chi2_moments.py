import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
def test_chi2_moments(self):
    N, df = (6, 15)
    cum = [_chi2_cumulant(n + 1, df) for n in range(N)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ne = ExpandedNormal(cum, name='edgw_chi2')
    assert_allclose([_chi2_moment(n, df) for n in range(N)], [ne.moment(n) for n in range(N)])
    check_pdf(ne, arg=(), msg='')
    check_cdf_ppf(ne, arg=(), msg='')
    check_cdf_sf(ne, arg=(), msg='')
    np.random.seed(765456)
    rvs = ne.rvs(size=500)
    check_distribution_rvs(ne, args=(), alpha=0.01, rvs=rvs)