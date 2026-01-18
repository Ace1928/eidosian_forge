from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
@vectorize(excluded=['Sa', 'abbr', 'octave', 'unicode'])
def midi_to_svara_h(midi: Union[_FloatLike_co, np.ndarray], *, Sa: _FloatLike_co, abbr: bool=True, octave: bool=True, unicode: bool=True) -> Union[str, np.ndarray]:
    """Convert MIDI numbers to Hindustani svara

    Parameters
    ----------
    midi : numeric or np.ndarray
        The MIDI number or numbers to convert

    Sa : number > 0
        MIDI number of the reference Sa.

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
        The svara corresponding to the given MIDI number(s)

    See Also
    --------
    hz_to_svara_h
    note_to_svara_h
    midi_to_svara_c
    midi_to_note

    Examples
    --------
    Convert a single midi number:

    >>> librosa.midi_to_svara_h(65, Sa=60)
    'm'

    The first three svara with Sa at midi number 60:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=60)
    array(['S', 'r', 'R'], dtype='<U1')

    With Sa=67, midi 60-62 are in the octave below:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=67)
    array(['ṃ', 'Ṃ', 'P̣'], dtype='<U2')

    Or without unicode decoration:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=67, unicode=False)
    array(['m,', 'M,', 'P,'], dtype='<U2')

    Or going up an octave, with Sa=60, and using unabbreviated notes

    >>> librosa.midi_to_svara_h([72, 73, 74], Sa=60, abbr=False)
    array(['Ṡa', 'ṙe', 'Ṙe'], dtype='<U3')
    """
    SVARA_MAP = ['Sa', 're', 'Re', 'ga', 'Ga', 'ma', 'Ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']
    SVARA_MAP_SHORT = list((s[0] for s in SVARA_MAP))
    svara_num = int(np.round(midi - Sa))
    if abbr:
        svara = SVARA_MAP_SHORT[svara_num % 12]
    else:
        svara = SVARA_MAP[svara_num % 12]
    if octave:
        if 24 > svara_num >= 12:
            if unicode:
                svara = svara[0] + '̇' + svara[1:]
            else:
                svara += "'"
        elif -12 <= svara_num < 0:
            if unicode:
                svara = svara[0] + '̣' + svara[1:]
            else:
                svara += ','
    return svara