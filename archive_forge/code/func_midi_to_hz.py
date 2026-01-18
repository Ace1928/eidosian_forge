from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def midi_to_hz(notes: _ScalarOrSequence[_FloatLike_co]) -> Union[np.ndarray, np.floating[Any]]:
    """Get the frequency (Hz) of MIDI note(s)

    Examples
    --------
    >>> librosa.midi_to_hz(36)
    65.406

    >>> librosa.midi_to_hz(np.arange(36, 48))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,
             87.307,   92.499,   97.999,  103.826,  110.   ,
            116.541,  123.471])

    Parameters
    ----------
    notes : int or np.ndarray [shape=(n,), dtype=int]
        midi number(s) of the note(s)

    Returns
    -------
    frequency : number or np.ndarray [shape=(n,), dtype=float]
        frequency (frequencies) of ``notes`` in Hz

    See Also
    --------
    hz_to_midi
    note_to_hz
    """
    return 440.0 * 2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0)