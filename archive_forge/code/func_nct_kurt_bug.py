import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.sppatch import expect_v2
from .distparams import distcont
def nct_kurt_bug():
    """test for incorrect kurtosis of nct

    D. Hogben, R. S. Pinkham, M. B. Wilk: The Moments of the Non-Central
    t-DistributionAuthor(s): Biometrika, Vol. 48, No. 3/4 (Dec., 1961),
    pp. 465-468
    """
    from numpy.testing import assert_almost_equal
    mvsk_10_1 = (1.08372, 1.325546, 0.39993, 1.2499424941142943)
    assert_almost_equal(stats.nct.stats(10, 1, moments='mvsk'), mvsk_10_1, decimal=6)
    c1 = np.array([1.08372])
    c2 = np.array([0.075546, 1.25])
    c3 = np.array([0.0297802, 0.580566])
    c4 = np.array([0.0425458, 1.17491, 6.25])
    nc = 1
    mc1 = c1.item()
    mc2 = (c2 * nc ** np.array([2, 0])).sum()
    mc3 = (c3 * nc ** np.array([3, 1])).sum()
    mc4 = c4 = np.array([0.0425458, 1.17491, 6.25])
    mvsk_nc = mc2mvsk((mc1, mc2, mc3, mc4))