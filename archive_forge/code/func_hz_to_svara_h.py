from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def hz_to_svara_h(frequencies: _ScalarOrSequence[_FloatLike_co], *, Sa: _FloatLike_co, abbr: bool=True, octave: bool=True, unicode: bool=True) -> Union[str, np.ndarray]:
    """Convert frequencies (in Hz) to Hindustani svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    frequencies : positive number or np.ndarray
        The frequencies (in Hz) to convert

    Sa : positive number
        Frequency (in Hz) of the reference Sa.

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
        The svara corresponding to the given frequency/frequencies

    See Also
    --------
    midi_to_svara_h
    note_to_svara_h
    hz_to_svara_c
    hz_to_note

    Examples
    --------
    Convert Sa in three octaves:

    >>> librosa.hz_to_svara_h([261/2, 261, 261*2], Sa=261)
    ['Ṣ', 'S', 'Ṡ']

    Convert one octave worth of frequencies with full names:

    >>> freqs = librosa.cqt_frequencies(n_bins=12, fmin=261)
    >>> librosa.hz_to_svara_h(freqs, Sa=freqs[0], abbr=False)
    ['Sa', 're', 'Re', 'ga', 'Ga', 'ma', 'Ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']
    """
    midis = hz_to_midi(frequencies)
    return midi_to_svara_h(midis, Sa=hz_to_midi(Sa), abbr=abbr, octave=octave, unicode=unicode)