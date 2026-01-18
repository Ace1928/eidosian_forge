import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
def expect_mc_bounds(dist, func=lambda x: 1, size=50000, lower=None, upper=None, conditional=False, overfact=1.2):
    """calculate expected value of function by Monte Carlo integration

    Parameters
    ----------
    dist : distribution instance
        needs to have rvs defined as a method for drawing random numbers
    func : callable
        function for which expectation is calculated, this function needs to
        be vectorized, integration is over axis=0
    size : int
        minimum number of random samples to use in the Monte Carlo integration,
        the actual number used can be larger because of oversampling.
    lower : None or array_like
        lower integration bounds, if None, then it is set to -inf
    upper : None or array_like
        upper integration bounds, if None, then it is set to +inf
    conditional : bool
        If true, then the expectation is conditional on being in within
        [lower, upper] bounds, otherwise it is unconditional
    overfact : float
        oversampling factor, the actual number of random variables drawn in
        each attempt are overfact * remaining draws. Extra draws are also
        used in the integration.


    Notes
    -----
    this does not batch

    Returns
    -------
    expected value : ndarray
        return of function func integrated over axis=0 by MonteCarlo, this will
        have the same shape as the return of func without axis=0

    Examples
    --------
    >>> mvn = mve.MVNormal([0,0],2.)
    >>> mve.expect_mc_bounds(mvn, lambda x: np.ones(x.shape[0]),
                                lower=[-10,-10],upper=[0,0])
    0.24990416666666668

    get 3 marginal moments with one integration

    >>> mvn = mve.MVNormal([0,0],1.)
    >>> mve.expect_mc_bounds(mvn, lambda x: np.dstack([x, x**2, x**3, x**4]),
        lower=[-np.inf,-np.inf], upper=[np.inf,np.inf])
    array([[  2.88629497e-03,   9.96706297e-01,  -2.51005344e-03,
              2.95240921e+00],
           [ -5.48020088e-03,   9.96004409e-01,  -2.23803072e-02,
              2.96289203e+00]])
    >>> from scipy import stats
    >>> [stats.norm.moment(i) for i in [1,2,3,4]]
    [0.0, 1.0, 0.0, 3.0]


    """
    rvsdim = dist.rvs(size=1).shape[-1]
    if lower is None:
        lower = -np.inf * np.ones(rvsdim)
    else:
        lower = np.asarray(lower)
    if upper is None:
        upper = np.inf * np.ones(rvsdim)
    else:
        upper = np.asarray(upper)

    def fun(x):
        return func(x)
    rvsli = []
    used = 0
    total = 0
    while True:
        remain = size - used
        rvs = dist.rvs(size=int(remain * overfact))
        total += int(size * overfact)
        rvsok = rvs[((rvs >= lower) & (rvs <= upper)).all(-1)]
        rvsok = np.atleast_2d(rvsok)
        used += rvsok.shape[0]
        rvsli.append(rvsok)
        print(used)
        if used >= size:
            break
    rvs = np.vstack(rvsli)
    print(rvs.shape)
    assert used == rvs.shape[0]
    mean_conditional = fun(rvs).mean(0)
    if conditional:
        return mean_conditional
    else:
        return mean_conditional * (used * 1.0 / total)