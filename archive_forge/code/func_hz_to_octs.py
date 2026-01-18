from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def hz_to_octs(frequencies: _ScalarOrSequence[_FloatLike_co], *, tuning: float=0.0, bins_per_octave: int=12) -> Union[np.floating[Any], np.ndarray]:
    """Convert frequencies (Hz) to (fractional) octave numbers.

    Examples
    --------
    >>> librosa.hz_to_octs(440.0)
    4.
    >>> librosa.hz_to_octs([32, 64, 128, 256])
    array([ 0.219,  1.219,  2.219,  3.219])

    Parameters
    ----------
    frequencies : number >0 or np.ndarray [shape=(n,)] or float
        scalar or vector of frequencies
    tuning : float
        Tuning deviation from A440 in (fractional) bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    octaves : number or np.ndarray [shape=(n,)]
        octave number for each frequency

    See Also
    --------
    octs_to_hz
    """
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    octs: np.ndarray = np.log2(np.asanyarray(frequencies) / (float(A440) / 16))
    return octs