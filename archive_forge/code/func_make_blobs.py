import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.random import sample_without_replacement
@validate_params({'n_samples': [Interval(Integral, 1, None, closed='left'), 'array-like'], 'n_features': [Interval(Integral, 1, None, closed='left')], 'centers': [Interval(Integral, 1, None, closed='left'), 'array-like', None], 'cluster_std': [Interval(Real, 0, None, closed='left'), 'array-like'], 'center_box': [tuple], 'shuffle': ['boolean'], 'random_state': ['random_state'], 'return_centers': ['boolean']}, prefer_skip_nested_validation=True)
def make_blobs(n_samples=100, n_features=2, *, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None, return_centers=False):
    """Generate isotropic Gaussian blobs for clustering.

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_datasets_plot_random_dataset.py`.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int or array-like, default=100
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.

        .. versionchanged:: v0.20
            one can now pass an array-like to the ``n_samples`` parameter

    n_features : int, default=2
        The number of features for each sample.

    centers : int or array-like of shape (n_centers, n_features), default=None
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.

    cluster_std : float or array-like of float, default=1.0
        The standard deviation of the clusters.

    center_box : tuple of float (min, max), default=(-10.0, 10.0)
        The bounding box for each cluster center when centers are
        generated at random.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_centers : bool, default=False
        If True, then return the centers of each cluster.

        .. versionadded:: 0.23

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.

    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster. Only returned if
        ``return_centers=True``.

    See Also
    --------
    make_classification : A more intricate variant.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
    >>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])
    """
    generator = check_random_state(random_state)
    if isinstance(n_samples, numbers.Integral):
        if centers is None:
            centers = 3
        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = generator.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]
    else:
        n_centers = len(n_samples)
        if centers is None:
            centers = generator.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        if not isinstance(centers, Iterable):
            raise ValueError('Parameter `centers` must be array-like. Got {!r} instead'.format(centers))
        if len(centers) != n_centers:
            raise ValueError(f'Length of `n_samples` not consistent with number of centers. Got n_samples = {n_samples} and centers = {centers}')
        centers = check_array(centers)
        n_features = centers.shape[1]
    if hasattr(cluster_std, '__len__') and len(cluster_std) != n_centers:
        raise ValueError('Length of `clusters_std` not consistent with number of centers. Got centers = {} and cluster_std = {}'.format(centers, cluster_std))
    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)
    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    cum_sum_n_samples = np.cumsum(n_samples_per_center)
    X = np.empty(shape=(sum(n_samples_per_center), n_features), dtype=np.float64)
    y = np.empty(shape=(sum(n_samples_per_center),), dtype=int)
    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        start_idx = cum_sum_n_samples[i - 1] if i > 0 else 0
        end_idx = cum_sum_n_samples[i]
        X[start_idx:end_idx] = generator.normal(loc=centers[i], scale=std, size=(n, n_features))
        y[start_idx:end_idx] = i
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)
    if return_centers:
        return (X, y, centers)
    else:
        return (X, y)