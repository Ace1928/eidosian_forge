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
def index_to_slice(idx: _SequenceLike[int], *, idx_min: Optional[int]=None, idx_max: Optional[int]=None, step: Optional[int]=None, pad: bool=True) -> List[slice]:
    """Generate a slice array from an index array.

    Parameters
    ----------
    idx : list-like
        Array of index boundaries
    idx_min, idx_max : None or int
        Minimum and maximum allowed indices
    step : None or int
        Step size for each slice.  If `None`, then the default
        step of 1 is used.
    pad : boolean
        If `True`, pad ``idx`` to span the range ``idx_min:idx_max``.

    Returns
    -------
    slices : list of slice
        ``slices[i] = slice(idx[i], idx[i+1], step)``
        Additional slice objects may be added at the beginning or end,
        depending on whether ``pad==True`` and the supplied values for
        ``idx_min`` and ``idx_max``.

    See Also
    --------
    fix_frames

    Examples
    --------
    >>> # Generate slices from spaced indices
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15))
    [slice(20, 35, None), slice(35, 50, None), slice(50, 65, None), slice(65, 80, None),
     slice(80, 95, None)]
    >>> # Pad to span the range (0, 100)
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100)
    [slice(0, 20, None), slice(20, 35, None), slice(35, 50, None), slice(50, 65, None),
     slice(65, 80, None), slice(80, 95, None), slice(95, 100, None)]
    >>> # Use a step of 5 for each slice
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100, step=5)
    [slice(0, 20, 5), slice(20, 35, 5), slice(35, 50, 5), slice(50, 65, 5), slice(65, 80, 5),
     slice(80, 95, 5), slice(95, 100, 5)]
    """
    idx_fixed = fix_frames(idx, x_min=idx_min, x_max=idx_max, pad=pad)
    return [slice(start, end, step) for start, end in zip(idx_fixed, idx_fixed[1:])]