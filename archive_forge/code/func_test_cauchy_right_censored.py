import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_cauchy_right_censored():
    """
    Test fitting the Cauchy distribution to right-censored data.

    Calculation in R, with two values not censored [1, 10] and
    one right-censored value [30].

    > library(fitdistrplus)
    > data <- data.frame(left=c(1, 10, 30), right=c(1, 10, NA))
    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' cauchy ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    location 7.100001
    scale    7.455866
    """
    data = CensoredData(uncensored=[1, 10], right=[30])
    loc, scale = cauchy.fit(data, optimizer=optimizer)
    assert_allclose(loc, 7.10001, rtol=5e-06)
    assert_allclose(scale, 7.455866, rtol=5e-06)