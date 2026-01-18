from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def note_to_midi(note: Union[str, _IterableLike[str], Iterable[str]], *, round_midi: bool=True) -> Union[float, np.ndarray]:
    """Convert one or more spelled notes to MIDI number(s).

    Notes may be spelled out with optional accidentals or octave numbers.

    The leading note name is case-insensitive.

    Sharps are indicated with ``#``, flats may be indicated with ``!`` or ``b``.

    Parameters
    ----------
    note : str or iterable of str
        One or more note names.
    round_midi : bool
        - If ``True``, midi numbers are rounded to the nearest integer.
        - If ``False``, allow fractional midi numbers.

    Returns
    -------
    midi : float or np.array
        Midi note numbers corresponding to inputs.

    Raises
    ------
    ParameterError
        If the input is not in valid note format

    See Also
    --------
    midi_to_note
    note_to_hz

    Examples
    --------
    >>> librosa.note_to_midi('C')
    12
    >>> librosa.note_to_midi('C#3')
    49
    >>> librosa.note_to_midi('C♯3')  # Using Unicode sharp
    49
    >>> librosa.note_to_midi('C♭3')  # Using Unicode flat
    47
    >>> librosa.note_to_midi('f4')
    65
    >>> librosa.note_to_midi('Bb-1')
    10
    >>> librosa.note_to_midi('A!8')
    116
    >>> librosa.note_to_midi('G𝄪6')  # Double-sharp
    93
    >>> librosa.note_to_midi('B𝄫6')  # Double-flat
    93
    >>> librosa.note_to_midi('C♭𝄫5')  # Triple-flats also work
    69
    >>> # Lists of notes also work
    >>> librosa.note_to_midi(['C', 'E', 'G'])
    array([12, 16, 19])
    """
    if not isinstance(note, str):
        return np.array([note_to_midi(n, round_midi=round_midi) for n in note])
    pitch_map: Dict[str, int] = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map: Dict[str, int] = {'#': 1, '': 0, 'b': -1, '!': -1, '♯': 1, '𝄪': 2, '♭': -1, '𝄫': -2, '♮': 0}
    match = notation.NOTE_RE.match(note)
    if not match:
        raise ParameterError(f'Improper note format: {note:s}')
    pitch = match.group('note').upper()
    offset = np.sum([acc_map[o] for o in match.group('accidental')])
    octave = match.group('octave')
    cents = match.group('cents')
    if not octave:
        octave = 0
    else:
        octave = int(octave)
    if not cents:
        cents = 0
    else:
        cents = int(cents) * 0.01
    note_value: float = 12 * (octave + 1) + pitch_map[pitch] + offset + cents
    if round_midi:
        return int(np.round(note_value))
    else:
        return note_value