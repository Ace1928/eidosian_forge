from scipy.stats import norm
from numpy import array, polyval, inf, asarray
def mackinnoncrit(N=1, regression='c', nobs=inf):
    """
    Returns the critical values for cointegrating and the ADF test.

    In 2010 MacKinnon updated the values of his 1994 paper with critical values
    for the augmented Dickey-Fuller tests.  These new values are to be
    preferred and are used here.

    Parameters
    ----------
    N : int
        The number of series of I(1) series for which the null of
        non-cointegration is being tested.  For N > 12, the critical values
        are linearly interpolated (not yet implemented).  For the ADF test,
        N = 1.
    reg : str {'c', 'tc', 'ctt', 'n'}
        Following MacKinnon (1996), these stand for the type of regression run.
        'c' for constant and no trend, 'tc' for constant with a linear trend,
        'ctt' for constant with a linear and quadratic trend, and 'n' for
        no constant.  The values for the no constant case are taken from the
        1996 paper, as they were not updated for 2010 due to the unrealistic
        assumptions that would underlie such a case.
    nobs : int or np.inf
        This is the sample size.  If the sample size is numpy.inf, then the
        asymptotic critical values are returned.

    References
    ----------
    .. [*] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions
        for Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.
    .. [*] MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics Working Papers 1227.
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    reg = regression
    if reg not in ['c', 'ct', 'n', 'ctt']:
        raise ValueError('regression keyword %s not understood' % reg)
    tau = tau_2010s[reg]
    if nobs is inf:
        return tau[N - 1, :, 0]
    else:
        val = tau[N - 1, :, ::-1]
        return polyval(val.T, 1.0 / nobs)