from statsmodels.compat.python import lmap
import numpy as np
def prob_bv_rectangle(lower, upper, cdf):
    """helper function for probability of a rectangle in a bivariate distribution

    Parameters
    ----------
    lower : array_like
        tuple of lower integration bounds
    upper : array_like
        tuple of upper integration bounds
    cdf : callable
        cdf(x,y), cumulative distribution function of bivariate distribution


    how does this generalize to more than 2 variates ?
    """
    probuu = cdf(*upper)
    probul = cdf(upper[0], lower[1])
    problu = cdf(lower[0], upper[1])
    probll = cdf(*lower)
    return probuu - probul - problu + probll