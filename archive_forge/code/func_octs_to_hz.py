from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def octs_to_hz(octs: _ScalarOrSequence[_FloatLike_co], *, tuning: float=0.0, bins_per_octave: int=12) -> Union[np.floating[Any], np.ndarray]:
    """Convert octaves numbers to frequencies.

    Octaves are counted relative to A.

    Examples
    --------
    >>> librosa.octs_to_hz(1)
    55.
    >>> librosa.octs_to_hz([-2, -1, 0, 1, 2])
    array([   6.875,   13.75 ,   27.5  ,   55.   ,  110.   ])

    Parameters
    ----------
    octs : np.ndarray [shape=(n,)] or float
        octave number for each frequency
    tuning : float
        Tuning deviation from A440 in (fractional) bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    frequencies : number or np.ndarray [shape=(n,)]
        scalar or vector of frequencies

    See Also
    --------
    hz_to_octs
    """
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    return float(A440) / 16 * 2.0 ** np.asanyarray(octs)