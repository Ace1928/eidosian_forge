import numpy as np
def halton(dim, n_sample, bounds=None, start_index=0):
    """Halton sequence.

    Pseudo-random number generator that generalize the Van der Corput sequence
    for multiple dimensions. Halton sequence use base-two Van der Corput
    sequence for the first dimension, base-three for its second and base-n for
    its n-dimension.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    n_sample : int
        Number of samples to generate in the parametr space.
    bounds : tuple or array_like ([min, k_vars], [max, k_vars])
        Desired range of transformed data. The transformation apply the bounds
        on the sample and not the theoretical space, unit cube. Thus min and
        max values of the sample will coincide with the bounds.
    start_index : int
        Index to start the sequence from.

    Returns
    -------
    sequence : array_like (n_samples, k_vars)
        Sequence of Halton.

    References
    ----------
    [1] Halton, "On the efficiency of certain quasi-random sequences of points
      in evaluating multi-dimensional integrals", Numerische Mathematik, 1960.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Halton.

    >>> from statsmodels.tools import sequences
    >>> sample = sequences.halton(dim=2, n_sample=5)

    Compute the quality of the sample using the discrepancy criterion.

    >>> uniformity = sequences.discrepancy(sample)

    If some wants to continue an existing design, extra points can be obtained.

    >>> sample_continued = sequences.halton(dim=2, n_sample=5, start_index=5)
    """
    base = n_primes(dim)
    sample = [van_der_corput(n_sample + 1, bdim, start_index) for bdim in base]
    sample = np.array(sample).T[1:]
    if bounds is not None:
        min_ = bounds.min(axis=0)
        max_ = bounds.max(axis=0)
        sample = sample * (max_ - min_) + min_
    return sample