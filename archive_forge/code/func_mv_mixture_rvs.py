import numpy as np
def mv_mixture_rvs(prob, size, dist, nvars, **kwargs):
    """
    Sample from a mixture of multivariate distributions.

    Parameters
    ----------
    prob : array_like
        Probability of sampling from each distribution in dist
    size : int
        The length of the returned sample.
    dist : array_like
        An iterable of distributions instances with callable method rvs.
    nvargs : int
        dimension of the multivariate distribution, could be inferred instead
    kwargs : tuple of dicts, optional
        ignored

    Examples
    --------
    Say we want 2000 random variables from mixture of normals with two
    multivariate normal distributions, and we want to sample from the
    first with probability .4 and the second with probability .6.

    import statsmodels.sandbox.distributions.mv_normal as mvd

    cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                       [ 0.5 ,  1.5 ,  0.6 ],
                       [ 0.75,  0.6 ,  2.  ]])

    mu = np.array([-1, 0.0, 2.0])
    mu2 = np.array([4, 2.0, 2.0])
    mvn3 = mvd.MVNormal(mu, cov3)
    mvn32 = mvd.MVNormal(mu2, cov3/2., 4)
    rvs = mix.mv_mixture_rvs([0.4, 0.6], 2000, [mvn3, mvn32], 3)
    """
    if len(prob) != len(dist):
        raise ValueError('You must provide as many probabilities as distributions')
    if not np.allclose(np.sum(prob), 1):
        raise ValueError('prob does not sum to 1')
    if kwargs is None:
        kwargs = ({},) * len(prob)
    idx = _make_index(prob, size)
    sample = np.empty((size, nvars))
    for i in range(len(prob)):
        sample_idx = idx[..., i]
        sample_size = sample_idx.sum()
        sample[sample_idx] = dist[i].rvs(size=int(sample_size))
    return sample