from decorator import decorator
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import sklearn
import sklearn.cluster
import sklearn.feature_extraction
import sklearn.neighbors
from ._cache import cache
from . import util
from .filters import diagonal_filter
from .util.exceptions import ParameterError
from typing import Any, Callable, Optional, TypeVar, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
@cache(level=30)
def recurrence_matrix(data: np.ndarray, *, k: Optional[int]=None, width: int=1, metric: str='euclidean', sym: bool=False, sparse: bool=False, mode: str='connectivity', bandwidth: Optional[Union[np.ndarray, _FloatLike_co, str]]=None, self: bool=False, axis: int=-1, full: bool=False) -> Union[np.ndarray, scipy.sparse.csc_matrix]:
    """Compute a recurrence matrix from a data matrix.

    ``rec[i, j]`` is non-zero if ``data[..., i]`` is a k-nearest neighbor
    of ``data[..., j]`` and ``|i - j| >= width``

    The specific value of ``rec[i, j]`` can have several forms, governed
    by the ``mode`` parameter below:

        - Connectivity: ``rec[i, j] = 1 or 0`` indicates that frames ``i`` and ``j`` are repetitions

        - Affinity: ``rec[i, j] > 0`` measures how similar frames ``i`` and ``j`` are.  This is also
          known as a (sparse) self-similarity matrix.

        - Distance: ``rec[i, j] > 0`` measures how distant frames ``i`` and ``j`` are.  This is also
          known as a (sparse) self-distance matrix.

    The general term *recurrence matrix* can refer to any of the three forms above.

    Parameters
    ----------
    data : np.ndarray [shape=(..., d, n)]
        A feature matrix.
        If the data has more than two dimensions (e.g., for multi-channel inputs),
        the leading dimensions are flattened prior to comparison.
        For example, a stereo input with shape `(2, d, n)` is
        automatically reshaped to `(2 * d, n)`.

    k : int > 0 [scalar] or None
        the number of nearest-neighbors for each sample

        Default: ``k = 2 * ceil(sqrt(t - 2 * width + 1))``,
        or ``k = 2`` if ``t <= 2 * width + 1``

    width : int >= 1 [scalar]
        only link neighbors ``(data[..., i], data[..., j])``
        if ``|i - j| >= width``

        ``width`` cannot exceed the length of the data.

    metric : str
        Distance metric to use for nearest-neighbor calculation.

        See `sklearn.neighbors.NearestNeighbors` for details.

    sym : bool [scalar]
        set ``sym=True`` to only link mutual nearest-neighbors

    sparse : bool [scalar]
        if False, returns a dense type (ndarray)
        if True, returns a sparse type (scipy.sparse.csc_matrix)

    mode : str, {'connectivity', 'distance', 'affinity'}
        If 'connectivity', a binary connectivity matrix is produced.

        If 'distance', then a non-zero entry contains the distance between
        points.

        If 'affinity', then non-zero entries are mapped to
        ``exp( - distance(i, j) / bandwidth)`` where ``bandwidth`` is
        as specified below.

    bandwidth : None, float > 0, ndarray, or str
        str options include ``{'med_k_scalar', 'mean_k', 'gmean_k', 'mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair'}``

        If ndarray is supplied, use ndarray as bandwidth for each i,j pair.

        If using ``mode='affinity'``, the ``bandwidth`` option can be used to set the
        bandwidth on the affinity kernel.

        If no value is provided or ``None``, default to ``'med_k_scalar'``.

        If ``bandwidth='med_k_scalar'``, a scalar bandwidth is set to the median distance
        of the k-th nearest neighbor for all samples.

        If ``bandwidth='mean_k'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        arithmetic mean between distances to the k-th nearest neighbor for sample i and sample j.

        If ``bandwidth='gmean_k'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        geometric mean between distances to the k-th nearest neighbor for sample i and j [#z]_.

        If ``bandwidth='mean_k_avg'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        arithmetic mean between the average distances to the first k-th nearest neighbors for
        sample i and sample j.
        This is similar to the approach in Wang et al. (2014) [#w]_ but does not include the distance
        between i and j.

        If ``bandwidth='gmean_k_avg'``, bandwidth is estimated for each sample-pair (i, j) by taking the
        geometric mean between the average distances to the first k-th nearest neighbors for
        sample i and sample j.

        If ``bandwidth='mean_k_avg_and_pair'``, bandwidth is estimated for each sample-pair (i, j) by
        taking the arithmetic mean between three terms: the average distances to the first
        k-th nearest neighbors for sample i and sample j respectively, as well as
        the distance between i and j.
        This is similar to the approach in Wang et al. (2014). [#w]_

        .. [#z] Zelnik-Manor, Lihi, and Pietro Perona. (2004).
            "Self-tuning spectral clustering." Advances in neural information processing systems 17.

        .. [#w] Wang, Bo, et al. (2014).
            "Similarity network fusion for aggregating data types on a genomic scale." Nat Methods 11, 333–337.
            https://doi.org/10.1038/nmeth.2810

    self : bool
        If ``True``, then the main diagonal is populated with self-links:
        0 if ``mode='distance'``, and 1 otherwise.

        If ``False``, the main diagonal is left empty.

    axis : int
        The axis along which to compute recurrence.
        By default, the last index (-1) is taken.

    full : bool
        If using ``mode ='affinity'`` or ``mode='distance'``, this option can be used to compute
        the full affinity or distance matrix as opposed a sparse matrix with only none-zero terms
        for the first k-neighbors of each sample.
        This option has no effect when using ``mode='connectivity'``.

        When using ``mode='distance'``, setting ``full=True`` will ignore ``k`` and ``width``.
        When using ``mode='affinity'``, setting ``full=True`` will use ``k`` exclusively for
        bandwidth estimation, and ignore ``width``.

    Returns
    -------
    rec : np.ndarray or scipy.sparse.csc_matrix, [shape=(t, t)]
        Recurrence matrix

    See Also
    --------
    sklearn.neighbors.NearestNeighbors
    scipy.spatial.distance.cdist
    librosa.feature.stack_memory
    recurrence_to_lag

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Find nearest neighbors in CQT space

    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> # Use time-delay embedding to get a cleaner recurrence matrix
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> R = librosa.segment.recurrence_matrix(chroma_stack)

    Or fix the number of nearest neighbors to 5

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, k=5)

    Suppress neighbors within +- 7 frames

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, width=7)

    Use cosine similarity instead of Euclidean distance

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, metric='cosine')

    Require mutual nearest neighbors

    >>> R = librosa.segment.recurrence_matrix(chroma_stack, sym=True)

    Use an affinity matrix instead of binary connectivity

    >>> R_aff = librosa.segment.recurrence_matrix(chroma_stack, metric='cosine',
    ...                                           mode='affinity')

    Plot the feature and recurrence matrices

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> imgsim = librosa.display.specshow(R, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, ax=ax[0])
    >>> ax[0].set(title='Binary recurrence (symmetric)')
    >>> imgaff = librosa.display.specshow(R_aff, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, cmap='magma_r', ax=ax[1])
    >>> ax[1].set(title='Affinity recurrence')
    >>> ax[1].label_outer()
    >>> fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
    >>> fig.colorbar(imgaff, ax=ax[1], orientation='horizontal')
    """
    data = np.atleast_2d(data)
    data = np.swapaxes(data, axis, 0)
    t = data.shape[0]
    data = data.reshape((t, -1), order='F')
    if width < 1 or width >= (t - 1) // 2:
        raise ParameterError('width={} must be at least 1 and at most (data.shape[{}] - 1) // 2={}'.format(width, axis, (t - 1) // 2))
    if mode not in ['connectivity', 'distance', 'affinity']:
        raise ParameterError(f"Invalid mode='{mode}'. Must be one of ['connectivity', 'distance', 'affinity']")
    if k is None:
        k = 2 * np.ceil(np.sqrt(t - 2 * width + 1))
    k = int(k)
    bandwidth_k = k
    if full and mode != 'connectivity':
        k = t
    try:
        knn = sklearn.neighbors.NearestNeighbors(n_neighbors=min(t - 1, k + 2 * width), metric=metric, algorithm='auto')
    except ValueError:
        knn = sklearn.neighbors.NearestNeighbors(n_neighbors=min(t - 1, k + 2 * width), metric=metric, algorithm='brute')
    knn.fit(data)
    if mode == 'affinity':
        kng_mode = 'distance'
    else:
        kng_mode = mode
    rec = knn.kneighbors_graph(mode=kng_mode).tolil()
    if not full:
        for diag in range(-width + 1, width):
            rec.setdiag(0, diag)
        for i in range(t):
            links = rec[i].nonzero()[1]
            idx = links[np.argsort(rec[i, links].toarray())][0]
            rec[i, idx[k:]] = 0
    if self:
        if mode == 'connectivity':
            rec.setdiag(1)
        elif mode == 'affinity':
            rec.setdiag(-1)
    else:
        rec.setdiag(0)
    if sym:
        rec = rec.minimum(rec.T)
    rec = rec.tocsr()
    rec.eliminate_zeros()
    if mode == 'connectivity':
        rec = rec.astype(bool)
    elif mode == 'affinity':
        rec.data[rec.data < 0] = 0.0
        aff_bandwidth = __affinity_bandwidth(rec, bandwidth, bandwidth_k)
        rec.data[:] = np.exp(rec.data / (-1 * aff_bandwidth))
    rec = rec.T
    if not sparse:
        rec = rec.toarray()
    return rec