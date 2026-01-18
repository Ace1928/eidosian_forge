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
def is_positive_int(x: float) -> bool:
    """Check that x is a positive integer, i.e. 1 or greater.

    Parameters
    ----------
    x : number

    Returns
    -------
    positive : bool
    """
    return isinstance(x, (int, np.integer)) and x > 0