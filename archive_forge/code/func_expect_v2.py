from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def expect_v2(self, fn=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False):
    """calculate expected value of a function with respect to the distribution

    location and scale only tested on a few examples

    Parameters
    ----------
        all parameters are keyword parameters
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        lb, ub : numbers
           lower and upper bound for integration, default is set using
           quantiles of the distribution, see Notes
        conditional : bool (False)
           If true then the integral is corrected by the conditional probability
           of the integration interval. The return value is the expectation
           of the function, conditional on being in the given interval.

    Returns
    -------
        expected value : float

    Notes
    -----
    This function has not been checked for it's behavior when the integral is
    not finite. The integration behavior is inherited from scipy.integrate.quad.

    The default limits are lb = self.ppf(1e-9, *args), ub = self.ppf(1-1e-9, *args)

    For some heavy tailed distributions, 'alpha', 'cauchy', 'halfcauchy',
    'levy', 'levy_l', and for 'ncf', the default limits are not set correctly
    even  when the expectation of the function is finite. In this case, the
    integration limits, lb and ub, should be chosen by the user. For example,
    for the ncf distribution, ub=1000 works in the examples.

    There are also problems with numerical integration in some other cases,
    for example if the distribution is very concentrated and the default limits
    are too large.

    """
    if fn is None:

        def fun(x, *args):
            return (loc + x * scale) * self._pdf(x, *args)
    else:

        def fun(x, *args):
            return fn(loc + x * scale) * self._pdf(x, *args)
    if lb is None:
        try:
            lb = self.ppf(1e-09, *args)
        except ValueError:
            lb = self.a
    else:
        lb = max(self.a, (lb - loc) / (1.0 * scale))
    if ub is None:
        try:
            ub = self.ppf(1 - 1e-09, *args)
        except ValueError:
            ub = self.b
    else:
        ub = min(self.b, (ub - loc) / (1.0 * scale))
    if conditional:
        invfac = self._sf(lb, *args) - self._sf(ub, *args)
    else:
        invfac = 1.0
    return integrate.quad(fun, lb, ub, args=args, limit=500)[0] / invfac