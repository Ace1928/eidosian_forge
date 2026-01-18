from k-means models and quantizing vectors by comparing them with
import warnings
import numpy as np
from collections import deque
from scipy._lib._array_api import (
from scipy._lib._util import check_random_state, rng_integers
from scipy.spatial.distance import cdist
from . import _vq
def kmeans(obs, k_or_guess, iter=20, thresh=1e-05, check_finite=True, *, seed=None):
    """
    Performs k-means on a set of observation vectors forming k clusters.

    The k-means algorithm adjusts the classification of the observations
    into clusters and updates the cluster centroids until the position of
    the centroids is stable over successive iterations. In this
    implementation of the algorithm, the stability of the centroids is
    determined by comparing the absolute value of the change in the average
    Euclidean distance between the observations and their corresponding
    centroids against a threshold. This yields
    a code book mapping centroids to codes and vice versa.

    Parameters
    ----------
    obs : ndarray
       Each row of the M by N array is an observation vector. The
       columns are the features seen during each observation.
       The features must be whitened first with the `whiten` function.

    k_or_guess : int or ndarray
       The number of centroids to generate. A code is assigned to
       each centroid, which is also the row index of the centroid
       in the code_book matrix generated.

       The initial k centroids are chosen by randomly selecting
       observations from the observation matrix. Alternatively,
       passing a k by N array specifies the initial k centroids.

    iter : int, optional
       The number of times to run k-means, returning the codebook
       with the lowest distortion. This argument is ignored if
       initial centroids are specified with an array for the
       ``k_or_guess`` parameter. This parameter does not represent the
       number of iterations of the k-means algorithm.

    thresh : float, optional
       Terminates the k-means algorithm if the change in
       distortion since the last k-means iteration is less than
       or equal to threshold.

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        Seed for initializing the pseudo-random number generator.
        If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        The default is None.

    Returns
    -------
    codebook : ndarray
       A k by N array of k centroids. The ith centroid
       codebook[i] is represented with the code i. The centroids
       and codes generated represent the lowest distortion seen,
       not necessarily the globally minimal distortion.
       Note that the number of centroids is not necessarily the same as the
       ``k_or_guess`` parameter, because centroids assigned to no observations
       are removed during iterations.

    distortion : float
       The mean (non-squared) Euclidean distance between the observations
       passed and the centroids generated. Note the difference to the standard
       definition of distortion in the context of the k-means algorithm, which
       is the sum of the squared distances.

    See Also
    --------
    kmeans2 : a different implementation of k-means clustering
       with more methods for generating initial centroids but without
       using a distortion change threshold as a stopping criterion.

    whiten : must be called prior to passing an observation matrix
       to kmeans.

    Notes
    -----
    For more functionalities or optimal performance, you can use
    `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    `This <https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html#comparison-of-high-performance-implementations>`_
    is a benchmark result of several implementations.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.vq import vq, kmeans, whiten
    >>> import matplotlib.pyplot as plt
    >>> features  = np.array([[ 1.9,2.3],
    ...                       [ 1.5,2.5],
    ...                       [ 0.8,0.6],
    ...                       [ 0.4,1.8],
    ...                       [ 0.1,0.1],
    ...                       [ 0.2,1.8],
    ...                       [ 2.0,0.5],
    ...                       [ 0.3,1.5],
    ...                       [ 1.0,1.0]])
    >>> whitened = whiten(features)
    >>> book = np.array((whitened[0],whitened[2]))
    >>> kmeans(whitened,book)
    (array([[ 2.3110306 ,  2.86287398],    # random
           [ 0.93218041,  1.24398691]]), 0.85684700941625547)

    >>> codes = 3
    >>> kmeans(whitened,codes)
    (array([[ 2.3110306 ,  2.86287398],    # random
           [ 1.32544402,  0.65607529],
           [ 0.40782893,  2.02786907]]), 0.5196582527686241)

    >>> # Create 50 datapoints in two clusters a and b
    >>> pts = 50
    >>> rng = np.random.default_rng()
    >>> a = rng.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
    >>> b = rng.multivariate_normal([30, 10],
    ...                             [[10, 2], [2, 1]],
    ...                             size=pts)
    >>> features = np.concatenate((a, b))
    >>> # Whiten data
    >>> whitened = whiten(features)
    >>> # Find 2 clusters in the data
    >>> codebook, distortion = kmeans(whitened, 2)
    >>> # Plot whitened data and cluster centers in red
    >>> plt.scatter(whitened[:, 0], whitened[:, 1])
    >>> plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    >>> plt.show()

    """
    xp = array_namespace(obs, k_or_guess)
    obs = as_xparray(obs, xp=xp, check_finite=check_finite)
    guess = as_xparray(k_or_guess, xp=xp, check_finite=check_finite)
    if iter < 1:
        raise ValueError('iter must be at least 1, got %s' % iter)
    if size(guess) != 1:
        if size(guess) < 1:
            raise ValueError('Asked for 0 clusters. Initial book was %s' % guess)
        return _kmeans(obs, guess, thresh=thresh, xp=xp)
    k = int(guess)
    if k != guess:
        raise ValueError('If k_or_guess is a scalar, it must be an integer.')
    if k < 1:
        raise ValueError('Asked for %d clusters.' % k)
    rng = check_random_state(seed)
    best_dist = xp.inf
    for i in range(iter):
        guess = _kpoints(obs, k, rng, xp)
        book, dist = _kmeans(obs, guess, thresh=thresh, xp=xp)
        if dist < best_dist:
            best_book = book
            best_dist = dist
    return (best_book, best_dist)