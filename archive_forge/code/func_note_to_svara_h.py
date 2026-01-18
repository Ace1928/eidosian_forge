from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def note_to_svara_h(notes: Union[str, _IterableLike[str]], *, Sa: str, abbr: bool=True, octave: bool=True, unicode: bool=True) -> Union[str, np.ndarray]:
    """Convert western notes to Hindustani svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    notes : str or iterable of str
        Notes to convert (e.g., `'C#'` or `['C4', 'Db4', 'D4']`

    Sa : str
        Note corresponding to Sa (e.g., `'C'` or `'C5'`).

        If no octave information is provided, it will default to octave 0
        (``C0`` ~= 16 Hz)

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'r', 'R', 'g', 'G', ...)

        If `False`, return long-form names ('Sa', 're', 'Re', 'ga', 'Ga', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or np.ndarray of str
        The svara corresponding to the given notes

    See Also
    --------
    midi_to_svara_h
    hz_to_svara_h
    note_to_svara_c
    note_to_midi
    note_to_hz

    Examples
    --------
    >>> librosa.note_to_svara_h(['C4', 'G4', 'C5', 'G5'], Sa='C5')
    ['Ṣ', 'P̣', 'S', 'P']
    """
    midis = note_to_midi(notes, round_midi=False)
    return midi_to_svara_h(midis, Sa=note_to_midi(Sa), abbr=abbr, octave=octave, unicode=unicode)