import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
def mela_to_degrees(mela: Union[str, int]) -> np.ndarray:
    """Construct the svara indices (degrees) for a given melakarta raga

    Parameters
    ----------
    mela : str or int
        Either the name or integer index ([1, 2, ..., 72]) of the melakarta raga

    Returns
    -------
    degrees : np.ndarray
        A list of the seven svara indices (starting from 0=Sa)
        contained in the specified raga

    See Also
    --------
    thaat_to_degrees
    key_to_degrees
    list_mela

    Examples
    --------
    Melakarta #1 (kanakangi):

    >>> librosa.mela_to_degrees(1)
    array([0, 1, 2, 5, 7, 8, 9])

    Or using a name directly:

    >>> librosa.mela_to_degrees('kanakangi')
    array([0, 1, 2, 5, 7, 8, 9])
    """
    if isinstance(mela, str):
        index = MELAKARTA_MAP[mela.lower()] - 1
    elif 0 < mela <= 72:
        index = mela - 1
    else:
        raise ParameterError(f'mela={mela} must be in range [1, 72]')
    degrees = [0]
    lower = index % 36
    if 0 <= lower < 6:
        degrees.extend([1, 2])
    elif 6 <= lower < 12:
        degrees.extend([1, 3])
    elif 12 <= lower < 18:
        degrees.extend([1, 4])
    elif 18 <= lower < 24:
        degrees.extend([2, 3])
    elif 24 <= lower < 30:
        degrees.extend([2, 4])
    else:
        degrees.extend([3, 4])
    if index < 36:
        degrees.append(5)
    else:
        degrees.append(6)
    degrees.append(7)
    upper = index % 6
    if upper == 0:
        degrees.extend([8, 9])
    elif upper == 1:
        degrees.extend([8, 10])
    elif upper == 2:
        degrees.extend([8, 11])
    elif upper == 3:
        degrees.extend([9, 10])
    elif upper == 4:
        degrees.extend([9, 11])
    else:
        degrees.extend([10, 11])
    return np.array(degrees)