from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from .util import pad_center, fill_off_diagonal, is_positive_int, tiny, expand_to
from .util.exceptions import ParameterError
from .filters import get_window
from typing import Any, Iterable, List, Optional, Tuple, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _IntLike_co
def dtw(X: Optional[np.ndarray]=None, Y: Optional[np.ndarray]=None, *, C: Optional[np.ndarray]=None, metric: str='euclidean', step_sizes_sigma: Optional[np.ndarray]=None, weights_add: Optional[np.ndarray]=None, weights_mul: Optional[np.ndarray]=None, subseq: bool=False, backtrack: bool=True, global_constraints: bool=False, band_rad: float=0.25, return_steps: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Dynamic time warping (DTW).

    This function performs a DTW and path backtracking on two sequences.
    We follow the nomenclature and algorithmic approach as described in [#]_.

    .. [#] Meinard Mueller
           Fundamentals of Music Processing — Audio, Analysis, Algorithms, Applications
           Springer Verlag, ISBN: 978-3-319-21944-8, 2015.

    Parameters
    ----------
    X : np.ndarray [shape=(..., K, N)]
        audio feature matrix (e.g., chroma features)

        If ``X`` has more than two dimensions (e.g., for multi-channel inputs), all leading
        dimensions are used when computing distance to ``Y``.

    Y : np.ndarray [shape=(..., K, M)]
        audio feature matrix (e.g., chroma features)

    C : np.ndarray [shape=(N, M)]
        Precomputed distance matrix. If supplied, X and Y must not be supplied and
        ``metric`` will be ignored.

    metric : str
        Identifier for the cost-function as documented
        in `scipy.spatial.distance.cdist()`

    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.

    weights_add : np.ndarray [shape=[n, ]]
        Additive weights to penalize certain step sizes.

    weights_mul : np.ndarray [shape=[n, ]]
        Multiplicative weights to penalize certain step sizes.

    subseq : bool
        Enable subsequence DTW, e.g., for retrieval tasks.

    backtrack : bool
        Enable backtracking in accumulated cost matrix.

    global_constraints : bool
        Applies global constraints to the cost matrix ``C`` (Sakoe-Chiba band).

    band_rad : float
        The Sakoe-Chiba band radius (1/2 of the width) will be
        ``int(radius*min(C.shape))``.

    return_steps : bool
        If true, the function returns ``steps``, the step matrix, containing
        the indices of the used steps from the cost accumulation step.

    Returns
    -------
    D : np.ndarray [shape=(N, M)]
        accumulated cost matrix.
        D[N, M] is the total alignment cost.
        When doing subsequence DTW, D[N,:] indicates a matching function.
    wp : np.ndarray [shape=(N, 2)]
        Warping path with index pairs.
        Each row of the array contains an index pair (n, m).
        Only returned when ``backtrack`` is True.
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
        Only returned when ``return_steps`` is True.

    Raises
    ------
    ParameterError
        If you are doing diagonal matching and Y is shorter than X or if an
        incompatible combination of X, Y, and C are supplied.

        If your input dimensions are incompatible.

        If the cost matrix has NaN values.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('brahms'), offset=10, duration=15)
    >>> X = librosa.feature.chroma_cens(y=y, sr=sr)
    >>> noise = np.random.rand(X.shape[0], 200)
    >>> Y = np.concatenate((noise, noise, X, noise), axis=1)
    >>> D, wp = librosa.sequence.dtw(X, Y, subseq=True)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
    ...                                ax=ax[0])
    >>> ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    >>> ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    >>> ax[0].legend()
    >>> fig.colorbar(img, ax=ax[0])
    >>> ax[1].plot(D[-1, :] / wp.shape[0])
    >>> ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
    ...           title='Matching cost function')
    """
    default_steps = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.uint32)
    default_weights_add = np.zeros(3, dtype=np.float64)
    default_weights_mul = np.ones(3, dtype=np.float64)
    if step_sizes_sigma is None:
        step_sizes_sigma = default_steps
        if weights_add is None:
            weights_add = default_weights_add
        if weights_mul is None:
            weights_mul = default_weights_mul
    else:
        if weights_add is None:
            weights_add = np.zeros(len(step_sizes_sigma), dtype=np.float64)
        if weights_mul is None:
            weights_mul = np.ones(len(step_sizes_sigma), dtype=np.float64)
        default_weights_add.fill(np.inf)
        default_weights_mul.fill(np.inf)
        step_sizes_sigma = np.concatenate((default_steps, step_sizes_sigma))
        weights_add = np.concatenate((default_weights_add, weights_add))
        weights_mul = np.concatenate((default_weights_mul, weights_mul))
    assert step_sizes_sigma is not None
    assert weights_add is not None
    assert weights_mul is not None
    if np.any(step_sizes_sigma < 0):
        raise ParameterError('step_sizes_sigma cannot contain negative values')
    if len(step_sizes_sigma) != len(weights_add):
        raise ParameterError('len(weights_add) must be equal to len(step_sizes_sigma)')
    if len(step_sizes_sigma) != len(weights_mul):
        raise ParameterError('len(weights_mul) must be equal to len(step_sizes_sigma)')
    if C is None and (X is None or Y is None):
        raise ParameterError('If C is not supplied, both X and Y must be supplied')
    if C is not None and (X is not None or Y is not None):
        raise ParameterError('If C is supplied, both X and Y must not be supplied')
    c_is_transposed = False
    C_local = False
    if C is None:
        C_local = True
        assert X is not None and Y is not None
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        X = np.swapaxes(X, -1, 0)
        Y = np.swapaxes(Y, -1, 0)
        X = X.reshape((X.shape[0], -1), order='F')
        Y = Y.reshape((Y.shape[0], -1), order='F')
        try:
            C = cdist(X, Y, metric=metric)
        except ValueError as exc:
            raise ParameterError('scipy.spatial.distance.cdist returned an error.\nPlease provide your input in the form X.shape=(K, N) and Y.shape=(K, M).\n 1-dimensional sequences should be reshaped to X.shape=(1, N) and Y.shape=(1, M).') from exc
        if subseq and X.shape[0] > Y.shape[0]:
            C = C.T
            c_is_transposed = True
    C = np.atleast_2d(C)
    if np.array_equal(step_sizes_sigma, np.array([[1, 1]])) and C.shape[0] > C.shape[1]:
        raise ParameterError('For diagonal matching: Y.shape[-1] >= X.shape[-11] (C.shape[1] >= C.shape[0])')
    max_0 = step_sizes_sigma[:, 0].max()
    max_1 = step_sizes_sigma[:, 1].max()
    if np.any(np.isnan(C)):
        raise ParameterError('DTW cost matrix C has NaN values. ')
    if global_constraints:
        if not C_local:
            C = np.copy(C)
        fill_off_diagonal(C, radius=band_rad, value=np.inf)
    D = np.ones(C.shape + np.array([max_0, max_1])) * np.inf
    D[max_0, max_1] = C[0, 0]
    if subseq:
        D[max_0, max_1:] = C[0, :]
    steps = np.zeros(D.shape, dtype=np.int32)
    steps[0, :] = 1
    steps[:, 0] = 2
    D: np.ndarray
    steps: np.ndarray
    D, steps = __dtw_calc_accu_cost(C, D, steps, step_sizes_sigma, weights_mul, weights_add, max_0, max_1)
    D = D[max_0:, max_1:]
    steps = steps[max_0:, max_1:]
    return_values: List[np.ndarray]
    if backtrack:
        wp: np.ndarray
        if subseq:
            if np.all(np.isinf(D[-1])):
                raise ParameterError('No valid sub-sequence warping path could be constructed with the given step sizes.')
            start = np.argmin(D[-1, :])
            _wp = __dtw_backtracking(steps, step_sizes_sigma, subseq, start)
        else:
            if np.isinf(D[-1, -1]):
                raise ParameterError('No valid sub-sequence warping path could be constructed with the given step sizes.')
            _wp = __dtw_backtracking(steps, step_sizes_sigma, subseq)
            if _wp[-1] != (0, 0):
                raise ParameterError('Unable to compute a full DTW warping path. You may want to try again with subseq=True.')
        wp = np.asarray(_wp, dtype=int)
        if subseq and (X is not None and Y is not None and (X.shape[0] > Y.shape[0]) or c_is_transposed or C.shape[0] > C.shape[1]):
            wp = np.fliplr(wp)
        return_values = [D, wp]
    else:
        return_values = [D]
    if return_steps:
        return_values.append(steps)
    if len(return_values) > 1:
        return tuple(return_values)
    else:
        return return_values[0]