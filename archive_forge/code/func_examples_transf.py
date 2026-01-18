import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.extras import (SkewNorm_gen, skewnorm,
from statsmodels.stats.moment_helpers import mc2mvsk, mnc2mc, mvsk2mnc
def examples_transf():
    print('Results for lognormal')
    lognormalg = ExpTransf_gen(stats.norm, a=0, name='Log transformed normal general')
    print(lognormalg.cdf(1))
    print(stats.lognorm.cdf(1, 1))
    print(lognormalg.stats())
    print(stats.lognorm.stats(1))
    print(lognormalg.rvs(size=5))
    print('Results for expgamma')
    loggammaexpg = LogTransf_gen(stats.gamma)
    print(loggammaexpg._cdf(1, 10))
    print(stats.loggamma.cdf(1, 10))
    print(loggammaexpg._cdf(2, 15))
    print(stats.loggamma.cdf(2, 15))
    print('Results for loglaplace')
    loglaplaceg = LogTransf_gen(stats.laplace)
    print(loglaplaceg._cdf(2))
    print(stats.loglaplace.cdf(2, 1))
    loglaplaceexpg = ExpTransf_gen(stats.laplace)
    print(loglaplaceexpg._cdf(2))
    stats.loglaplace.cdf(3, 3)
    loglaplaceexpg._cdf(3, 0, 1.0 / 3)