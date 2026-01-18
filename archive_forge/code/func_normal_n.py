from ...context import current_context
from ..numpy import _internal as _npi
def normal_n(loc=0.0, scale=1.0, batch_shape=None, dtype=None, ctx=None):
    """Draw random samples from a normal (Gaussian) distribution.

    Samples are distributed according to a normal distribution parametrized
    by *loc* (mean) and *scale* (standard deviation).


    Parameters
    ----------
    loc : float, optional
        Mean (centre) of the distribution.
    scale : float, optional
        Standard deviation (spread or "width") of the distribution.
    batch_shape : int or tuple of ints, optional
        Batch shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k * broadcast(low, high).size`` samples are drawn.
        If size is ``None`` (default),
        a scalar tensor containing a single value is returned if
        ``low`` and ``high`` are both scalars. Otherwise,
        ``np.broadcast(loc, scale).size`` samples are drawn.
    dtype : {'float16', 'float32', 'float64'}, optional
        Data type of output samples. Default is 'float32'
    ctx : Context, optional
        Device context of output, default is current context.

    Returns
    -------
    out : ndarray
        Drawn samples from the parameterized normal distribution.

    Notes
    -----
    The probability density for the Gaussian distribution is

    .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                     e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

    where :math:`\\mu` is the mean and :math:`\\sigma` the standard
    deviation. The square of the standard deviation, :math:`\\sigma^2`,
    is called the variance.

    The function has its peak at the mean, and its "spread" increases with
    the standard deviation (the function reaches 0.607 times its maximum at
    :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
    `numpy.random.normal` is more likely to return samples lying close to
    the mean, rather than those far away.

    References
    ----------
    .. [1] Wikipedia, "Normal distribution",
           https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
           Random Variables and Random Signal Principles", 4th ed., 2001,
           pp. 51, 51, 125.

    Examples
    --------
    >>> mu, sigma = 0, 0.1 # mean and standard deviation
    >>> s = np.random.normal(mu, sigma, 1000)

    Verify the mean and the variance:

    >>> np.abs(mu - np.mean(s)) < 0.01
    array(True)
    """
    from ...numpy import ndarray as np_ndarray
    input_type = (isinstance(loc, np_ndarray), isinstance(scale, np_ndarray))
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if batch_shape == ():
        batch_shape = None
    if input_type == (True, True):
        return _npi.normal_n(loc, scale, loc=None, scale=None, size=batch_shape, ctx=ctx, dtype=dtype)
    elif input_type == (False, True):
        return _npi.normal_n(scale, loc=loc, scale=None, size=batch_shape, ctx=ctx, dtype=dtype)
    elif input_type == (True, False):
        return _npi.normal_n(loc, loc=None, scale=scale, size=batch_shape, ctx=ctx, dtype=dtype)
    else:
        return _npi.normal_n(loc=loc, scale=scale, size=batch_shape, ctx=ctx, dtype=dtype)