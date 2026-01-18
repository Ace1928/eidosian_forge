import warnings
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, _fit_context
from ..isotonic import IsotonicRegression
from ..metrics import euclidean_distances
from ..utils import check_array, check_random_state, check_symmetric
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
@validate_params({'dissimilarities': ['array-like'], 'metric': ['boolean'], 'n_components': [Interval(Integral, 1, None, closed='left')], 'init': ['array-like', None], 'n_init': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'verbose': ['verbose'], 'eps': [Interval(Real, 0, None, closed='left')], 'random_state': ['random_state'], 'return_n_iter': ['boolean'], 'normalized_stress': ['boolean', StrOptions({'auto'})]}, prefer_skip_nested_validation=True)
def smacof(dissimilarities, *, metric=True, n_components=2, init=None, n_init=8, n_jobs=None, max_iter=300, verbose=0, eps=0.001, random_state=None, return_n_iter=False, normalized_stress='auto'):
    """Compute multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : array-like of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : array-like of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, default=8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    normalized_stress : bool or "auto" default="auto"
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.

        .. versionadded:: 1.2

        .. versionchanged:: 1.4
           The default value changed from `False` to `"auto"` in version 1.4.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import smacof
    >>> from sklearn.metrics import euclidean_distances
    >>> X = np.array([[0, 1, 2], [1, 0, 3],[2, 3, 0]])
    >>> dissimilarities = euclidean_distances(X)
    >>> mds_result, stress = smacof(dissimilarities, n_components=2, random_state=42)
    >>> mds_result
    array([[ 0.05... -1.07... ],
           [ 1.74..., -0.75...],
           [-1.79...,  1.83...]])
    >>> stress
    0.0012...
    """
    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)
    if normalized_stress == 'auto':
        normalized_stress = not metric
    if normalized_stress and metric:
        raise ValueError('Normalized stress is not supported for metric MDS. Either set `normalized_stress=False` or use `metric=False`.')
    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn('Explicit initial positions passed: performing only one init of the MDS instead of %d' % n_init)
            n_init = 1
    best_pos, best_stress = (None, None)
    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(dissimilarities, metric=metric, n_components=n_components, init=init, max_iter=max_iter, verbose=verbose, eps=eps, random_state=random_state, normalized_stress=normalized_stress)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))((delayed(_smacof_single)(dissimilarities, metric=metric, n_components=n_components, init=init, max_iter=max_iter, verbose=verbose, eps=eps, random_state=seed, normalized_stress=normalized_stress) for seed in seeds))
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]
    if return_n_iter:
        return (best_pos, best_stress, best_iter)
    else:
        return (best_pos, best_stress)