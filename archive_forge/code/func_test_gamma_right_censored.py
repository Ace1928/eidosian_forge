import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_gamma_right_censored():
    """
    Fit gamma shape and scale to data with one right-censored value.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, 25.0),
    +                    right=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, NA))
    > result = fitdistcens(data, 'gamma', start=list(shape=1, scale=10),
    +                      control=list(reltol=1e-13))
    > result
    Fitting of the distribution ' gamma ' on censored data by maximum
      likelihood
    Parameters:
          estimate
    shape 1.447623
    scale 8.360197
    > result$sd
        shape     scale
    0.7053086 5.1016531
    """
    x = CensoredData.right_censored([2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, 25.0], [0] * 7 + [1])
    a, loc, scale = gamma.fit(x, floc=0, optimizer=optimizer)
    assert_allclose(a, 1.447623, rtol=5e-06)
    assert loc == 0
    assert_allclose(scale, 8.360197, rtol=5e-06)