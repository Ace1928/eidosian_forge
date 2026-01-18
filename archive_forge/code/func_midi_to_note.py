from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
@vectorize(excluded=['octave', 'cents', 'key', 'unicode'])
def midi_to_note(midi: _ScalarOrSequence[_FloatLike_co], *, octave: bool=True, cents: bool=False, key: str='C:maj', unicode: bool=True) -> Union[str, np.ndarray]:
    """Convert one or more MIDI numbers to note strings.

    MIDI numbers will be rounded to the nearest integer.

    Notes will be of the format 'C0', 'C♯0', 'D0', ...

    Examples
    --------
    >>> librosa.midi_to_note(0)
    'C-1'

    >>> librosa.midi_to_note(37)
    'C♯2'

    >>> librosa.midi_to_note(37, unicode=False)
    'C#2'

    >>> librosa.midi_to_note(-2)
    'A♯-2'

    >>> librosa.midi_to_note(104.7)
    'A7'

    >>> librosa.midi_to_note(104.7, cents=True)
    'A7-30'

    >>> librosa.midi_to_note(np.arange(12, 24)))
    array(['C0', 'C♯0', 'D0', 'D♯0', 'E0', 'F0', 'F♯0', 'G0', 'G♯0', 'A0',
           'A♯0', 'B0'], dtype='<U3')

    Use a key signature to resolve enharmonic equivalences

    >>> librosa.midi_to_note(range(12, 24), key='F:min')
    array(['C0', 'D♭0', 'D0', 'E♭0', 'E0', 'F0', 'G♭0', 'G0', 'A♭0', 'A0',
           'B♭0', 'B0'], dtype='<U3')

    Parameters
    ----------
    midi : int or iterable of int
        Midi numbers to convert.

    octave : bool
        If True, include the octave number

    cents : bool
        If true, cent markers will be appended for fractional notes.
        Eg, ``midi_to_note(69.3, cents=True) == 'A4+03'``

    key : str
        A key signature to use when resolving enharmonic equivalences.

    unicode : bool
        If ``True`` (default), accidentals will use Unicode notation: ♭ or ♯

        If ``False``, accidentals will use ASCII-compatible notation: b or #

    Returns
    -------
    notes : str or np.ndarray of str
        Strings describing each midi note.

    Raises
    ------
    ParameterError
        if ``cents`` is True and ``octave`` is False

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    key_to_notes
    """
    if cents and (not octave):
        raise ParameterError('Cannot encode cents without octave information.')
    note_map = notation.key_to_notes(key=key, unicode=unicode)
    note_num = int(np.round(midi))
    note_cents = int(100 * np.around(midi - note_num, 2))
    note = note_map[note_num % 12]
    if octave:
        note = '{:s}{:0d}'.format(note, int(note_num / 12) - 1)
    if cents:
        note = f'{note:s}{note_cents:+02d}'
    return note