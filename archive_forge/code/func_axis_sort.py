from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def axis_sort(S: np.ndarray, *, axis: int=-1, index: bool=False, value: Optional[Callable[..., Any]]=None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Sort an array along its rows or columns.

    Examples
    --------
    Visualize NMF output for a spectrogram S

    >>> # Sort the columns of W by peak frequency bin
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> W, H = librosa.decompose.decompose(S, n_components=64)
    >>> W_sort = librosa.util.axis_sort(W)

    Or sort by the lowest frequency bin

    >>> W_sort = librosa.util.axis_sort(W, value=np.argmin)

    Or sort the rows instead of the columns

    >>> W_sort_rows = librosa.util.axis_sort(W, axis=0)

    Get the sorting index also, and use it to permute the rows of H

    >>> W_sort, idx = librosa.util.axis_sort(W, index=True)
    >>> H_sort = H[idx, :]

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2)
    >>> img_w = librosa.display.specshow(librosa.amplitude_to_db(W, ref=np.max),
    ...                                  y_axis='log', ax=ax[0, 0])
    >>> ax[0, 0].set(title='W')
    >>> ax[0, 0].label_outer()
    >>> img_act = librosa.display.specshow(H, x_axis='time', ax=ax[0, 1])
    >>> ax[0, 1].set(title='H')
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(W_sort,
    ...                                                  ref=np.max),
    ...                          y_axis='log', ax=ax[1, 0])
    >>> ax[1, 0].set(title='W sorted')
    >>> librosa.display.specshow(H_sort, x_axis='time', ax=ax[1, 1])
    >>> ax[1, 1].set(title='H sorted')
    >>> ax[1, 1].label_outer()
    >>> fig.colorbar(img_w, ax=ax[:, 0], orientation='horizontal')
    >>> fig.colorbar(img_act, ax=ax[:, 1], orientation='horizontal')

    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        Array to be sorted

    axis : int [scalar]
        The axis along which to compute the sorting values

        - ``axis=0`` to sort rows by peak column index
        - ``axis=1`` to sort columns by peak row index

    index : boolean [scalar]
        If true, returns the index array as well as the permuted data.

    value : function
        function to return the index corresponding to the sort order.
        Default: `np.argmax`.

    Returns
    -------
    S_sort : np.ndarray [shape=(d, n)]
        ``S`` with the columns or rows permuted in sorting order
    idx : np.ndarray (optional) [shape=(d,) or (n,)]
        If ``index == True``, the sorting index used to permute ``S``.
        Length of ``idx`` corresponds to the selected ``axis``.

    Raises
    ------
    ParameterError
        If ``S`` does not have exactly 2 dimensions (``S.ndim != 2``)
    """
    if value is None:
        value = np.argmax
    if S.ndim != 2:
        raise ParameterError('axis_sort is only defined for 2D arrays')
    bin_idx = value(S, axis=np.mod(1 - axis, S.ndim))
    idx = np.argsort(bin_idx)
    sort_slice = [slice(None)] * S.ndim
    sort_slice[axis] = idx
    if index:
        return (S[tuple(sort_slice)], idx)
    else:
        return S[tuple(sort_slice)]