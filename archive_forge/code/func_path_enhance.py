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
def path_enhance(R: np.ndarray, n: int, *, window: _WindowSpec='hann', max_ratio: float=2.0, min_ratio: Optional[float]=None, n_filters: int=7, zero_mean: bool=False, clip: bool=True, **kwargs: Any) -> np.ndarray:
    """Multi-angle path enhancement for self- and cross-similarity matrices.

    This function convolves multiple diagonal smoothing filters with a self-similarity (or
    recurrence) matrix R, and aggregates the result by an element-wise maximum.

    Technically, the output is a matrix R_smooth such that::

        R_smooth[i, j] = max_theta (R * filter_theta)[i, j]

    where `*` denotes 2-dimensional convolution, and ``filter_theta`` is a smoothing filter at
    orientation theta.

    This is intended to provide coherent temporal smoothing of self-similarity matrices
    when there are changes in tempo.

    Smoothing filters are generated at evenly spaced orientations between min_ratio and
    max_ratio.

    This function is inspired by the multi-angle path enhancement of [#]_, but differs by
    modeling tempo differences in the space of similarity matrices rather than re-sampling
    the underlying features prior to generating the self-similarity matrix.

    .. [#] Müller, Meinard and Frank Kurth.
            "Enhancing similarity matrices for music audio analysis."
            2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings.
            Vol. 5. IEEE, 2006.

    .. note:: if using recurrence_matrix to construct the input similarity matrix, be sure to include the main
              diagonal by setting ``self=True``.  Otherwise, the diagonal will be suppressed, and this is likely to
              produce discontinuities which will pollute the smoothing filter response.

    Parameters
    ----------
    R : np.ndarray
        The self- or cross-similarity matrix to be smoothed.
        Note: sparse inputs are not supported.

        If the recurrence matrix is multi-dimensional, e.g. `shape=(c, n, n)`,
        then enhancement is conducted independently for each leading channel.

    n : int > 0
        The length of the smoothing filter

    window : window specification
        The type of smoothing filter to use.  See `filters.get_window` for more information
        on window specification formats.

    max_ratio : float > 0
        The maximum tempo ratio to support

    min_ratio : float > 0
        The minimum tempo ratio to support.
        If not provided, it will default to ``1/max_ratio``

    n_filters : int >= 1
        The number of different smoothing filters to use, evenly spaced
        between ``min_ratio`` and ``max_ratio``.

        If ``min_ratio = 1/max_ratio`` (the default), using an odd number
        of filters will ensure that the main diagonal (ratio=1) is included.

    zero_mean : bool
        By default, the smoothing filters are non-negative and sum to one (i.e. are averaging
        filters).

        If ``zero_mean=True``, then the smoothing filters are made to sum to zero by subtracting
        a constant value from the non-diagonal coordinates of the filter.  This is primarily
        useful for suppressing blocks while enhancing diagonals.

    clip : bool
        If True, the smoothed similarity matrix will be thresholded at 0, and will not contain
        negative entries.

    **kwargs : additional keyword arguments
        Additional arguments to pass to `scipy.ndimage.convolve`

    Returns
    -------
    R_smooth : np.ndarray, shape=R.shape
        The smoothed self- or cross-similarity matrix

    See Also
    --------
    librosa.filters.diagonal_filter
    recurrence_matrix

    Examples
    --------
    Use a 51-frame diagonal smoothing filter to enhance paths in a recurrence matrix

    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 2048
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> rec = librosa.segment.recurrence_matrix(chroma_stack, mode='affinity', self=True)
    >>> rec_smooth = librosa.segment.path_enhance(rec, 51, window='hann', n_filters=7)

    Plot the recurrence matrix before and after smoothing

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(rec, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, ax=ax[0])
    >>> ax[0].set(title='Unfiltered recurrence')
    >>> imgpe = librosa.display.specshow(rec_smooth, x_axis='s', y_axis='s',
    ...                          hop_length=hop_length, ax=ax[1])
    >>> ax[1].set(title='Multi-angle enhanced recurrence')
    >>> ax[1].label_outer()
    >>> fig.colorbar(img, ax=ax[0], orientation='horizontal')
    >>> fig.colorbar(imgpe, ax=ax[1], orientation='horizontal')
    """
    if min_ratio is None:
        min_ratio = 1.0 / max_ratio
    elif min_ratio > max_ratio:
        raise ParameterError(f'min_ratio={min_ratio} cannot exceed max_ratio={max_ratio}')
    R_smooth = None
    for ratio in np.logspace(np.log2(min_ratio), np.log2(max_ratio), num=n_filters, base=2):
        kernel = diagonal_filter(window, n, slope=ratio, zero_mean=zero_mean)
        shape = [1] * R.ndim
        shape[-2:] = kernel.shape
        kernel = np.reshape(kernel, shape)
        if R_smooth is None:
            R_smooth = scipy.ndimage.convolve(R, kernel, **kwargs)
        else:
            np.maximum(R_smooth, scipy.ndimage.convolve(R, kernel, **kwargs), out=R_smooth)
    if clip:
        np.clip(R_smooth, 0, None, out=R_smooth)
    return np.asanyarray(R_smooth)