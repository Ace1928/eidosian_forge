import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_nct():
    """
    Test fitting the noncentral t distribution to censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(1, 2, 3, 5, 8, 10, 25, 25),
    +                    right=c(1, 2, 3, 5, 8, 10, NA, NA))
    > result = fitdistcens(data, 't', control=list(reltol=1e-14),
    +                      start=list(df=1, ncp=2))
    > result
    Fitting of the distribution ' t ' on censored data by maximum likelihood
    Parameters:
         estimate
    df  0.5432336
    ncp 2.8893565

    """
    data = CensoredData.right_censored([1, 2, 3, 5, 8, 10, 25, 25], [0, 0, 0, 0, 0, 0, 1, 1])
    with np.errstate(over='ignore'):
        df, nc, loc, scale = nct.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(df, 0.5432336, rtol=5e-06)
    assert_allclose(nc, 2.8893565, rtol=5e-06)
    assert loc == 0
    assert scale == 1