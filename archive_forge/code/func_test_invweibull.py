import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_invweibull():
    """
    Fit invweibull to censored data.

    Here is the calculation in R.  The 'frechet' distribution from the evd
    package matches SciPy's invweibull distribution.  The `loc` parameter
    is fixed at 0.

    > library(evd)
    > library(fitdistrplus)
    > data = data.frame(left=c(0, 2, 3, 9, 10, 10),
    +                   right=c(1, 2, 3, 9, NA, NA))
    > result = fitdistcens(data, 'frechet',
    +                      control=list(reltol=1e-14),
    +                      start=list(loc=4, scale=5))
    > result
    Fitting of the distribution ' frechet ' on censored data by maximum
    likelihood
    Parameters:
           estimate
    scale 2.7902200
    shape 0.6379845
    Fixed parameters:
        value
    loc     0
    """
    data = CensoredData(uncensored=[2, 3, 9], right=[10, 10], interval=[[0, 1]])
    c, loc, scale = invweibull.fit(data, floc=0, optimizer=optimizer)
    assert_allclose(c, 0.6379845, rtol=5e-06)
    assert loc == 0
    assert_allclose(scale, 2.79022, rtol=5e-06)