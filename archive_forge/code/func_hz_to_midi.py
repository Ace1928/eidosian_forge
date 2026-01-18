from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def hz_to_midi(frequencies: _ScalarOrSequence[_FloatLike_co]) -> Union[np.ndarray, np.floating[Any]]:
    """Get MIDI note number(s) for given frequencies

    Examples
    --------
    >>> librosa.hz_to_midi(60)
    34.506
    >>> librosa.hz_to_midi([110, 220, 440])
    array([ 45.,  57.,  69.])

    Parameters
    ----------
    frequencies : float or np.ndarray [shape=(n,), dtype=float]
        frequencies to convert

    Returns
    -------
    note_nums : number or np.ndarray [shape=(n,), dtype=float]
        MIDI notes to ``frequencies``

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    """
    midi: np.ndarray = 12 * (np.log2(np.asanyarray(frequencies)) - np.log2(440.0)) + 69
    return midi