import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def corner_peaks(image, min_distance=1, threshold_abs=None, threshold_rel=None, exclude_border=True, indices=True, num_peaks=np.inf, footprint=None, labels=None, *, num_peaks_per_label=np.inf, p_norm=np.inf):
    """Find peaks in corner measure response image.

    This differs from `skimage.feature.peak_local_max` in that it suppresses
    multiple connected peaks with the same accumulator value.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks.
    * : *
        See :py:meth:`skimage.feature.peak_local_max`.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    See also
    --------
    skimage.feature.peak_local_max

    Notes
    -----
    .. versionchanged:: 0.18
        The default value of `threshold_rel` has changed to None, which
        corresponds to letting `skimage.feature.peak_local_max` decide on the
        default. This is equivalent to `threshold_rel=0`.

    The `num_peaks` limit is applied before suppression of connected peaks.
    To limit the number of peaks after suppression, set `num_peaks=np.inf` and
    post-process the output of this function.

    Examples
    --------
    >>> from skimage.feature import peak_local_max
    >>> response = np.zeros((5, 5))
    >>> response[2:4, 2:4] = 1
    >>> response
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 0., 0.]])
    >>> peak_local_max(response)
    array([[2, 2],
           [2, 3],
           [3, 2],
           [3, 3]])
    >>> corner_peaks(response)
    array([[2, 2]])

    """
    if np.isinf(num_peaks):
        num_peaks = None
    coords = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold_abs, threshold_rel=threshold_rel, exclude_border=exclude_border, num_peaks=np.inf, footprint=footprint, labels=labels, num_peaks_per_label=num_peaks_per_label)
    if len(coords):
        tree = spatial.cKDTree(coords)
        rejected_peaks_indices = set()
        for idx, point in enumerate(coords):
            if idx not in rejected_peaks_indices:
                candidates = tree.query_ball_point(point, r=min_distance, p=p_norm)
                candidates.remove(idx)
                rejected_peaks_indices.update(candidates)
        coords = np.delete(coords, tuple(rejected_peaks_indices), axis=0)[:num_peaks]
    if indices:
        return coords
    peaks = np.zeros_like(image, dtype=bool)
    peaks[tuple(coords.T)] = True
    return peaks