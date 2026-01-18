import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.extras import (SkewNorm_gen, skewnorm,
from statsmodels.stats.moment_helpers import mc2mvsk, mnc2mc, mvsk2mnc
def example_T():
    skewt = ACSkewT_gen()
    rvs = skewt.rvs(10, 0, size=500)
    print('sample mean var: ', rvs.mean(), rvs.var())
    print('theoretical mean var', skewt.stats(10, 0))
    print('t mean var', stats.t.stats(10))
    print(skewt.stats(10, 1000))
    rvs = np.abs(stats.t.rvs(10, size=1000))
    print(rvs.mean(), rvs.var())