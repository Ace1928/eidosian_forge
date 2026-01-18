import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
@cache(level=10)
def key_to_notes(key: str, *, unicode: bool=True) -> List[str]:
    """List all 12 note names in the chromatic scale, as spelled according to
    a given key (major or minor).

    This function exists to resolve enharmonic equivalences between different
    spellings for the same pitch (e.g. C♯ vs D♭), and is primarily useful when producing
    human-readable outputs (e.g. plotting) for pitch content.

    Note names are decided by the following rules:

    1. If the tonic of the key has an accidental (sharp or flat), that accidental will be
       used consistently for all notes.

    2. If the tonic does not have an accidental, accidentals will be inferred to minimize
       the total number used for diatonic scale degrees.

    3. If there is a tie (e.g., in the case of C:maj vs A:min), sharps will be preferred.

    Parameters
    ----------
    key : string
        Must be in the form TONIC:key.  Tonic must be upper case (``CDEFGAB``),
        key must be lower-case (``maj`` or ``min``).

        Single accidentals (``b!♭`` for flat, or ``#♯`` for sharp) are supported.

        Examples: ``C:maj, Db:min, A♭:min``.

    unicode : bool
        If ``True`` (default), use Unicode symbols (♯𝄪♭𝄫)for accidentals.

        If ``False``, Unicode symbols will be mapped to low-order ASCII representations::

            ♯ -> #, 𝄪 -> ##, ♭ -> b, 𝄫 -> bb

    Returns
    -------
    notes : list
        ``notes[k]`` is the name for semitone ``k`` (starting from C)
        under the given key.  All chromatic notes (0 through 11) are
        included.

    See Also
    --------
    midi_to_note

    Examples
    --------
    `C:maj` will use all sharps

    >>> librosa.key_to_notes('C:maj')
    ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

    `A:min` has the same notes

    >>> librosa.key_to_notes('A:min')
    ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

    `A♯:min` will use sharps, but spell note 0 (`C`) as `B♯`

    >>> librosa.key_to_notes('A#:min')
    ['B♯', 'C♯', 'D', 'D♯', 'E', 'E♯', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

    `G♯:maj` will use a double-sharp to spell note 7 (`G`) as `F𝄪`:

    >>> librosa.key_to_notes('G#:maj')
    ['B♯', 'C♯', 'D', 'D♯', 'E', 'E♯', 'F♯', 'F𝄪', 'G♯', 'A', 'A♯', 'B']

    `F♭:min` will use double-flats

    >>> librosa.key_to_notes('Fb:min')
    ['D𝄫', 'D♭', 'E𝄫', 'E♭', 'F♭', 'F', 'G♭', 'A𝄫', 'A♭', 'B𝄫', 'B♭', 'C♭']
    """
    match = KEY_RE.match(key)
    if not match:
        raise ParameterError(f'Improper key format: {key:s}')
    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1, '♯': 1, '♭': -1}
    tonic = match.group('tonic').upper()
    accidental = match.group('accidental')
    offset = acc_map[accidental]
    scale = match.group('scale')[:3].lower()
    major = scale == 'maj'
    if major:
        tonic_number = (pitch_map[tonic] + offset) * 7 % 12
    else:
        tonic_number = ((pitch_map[tonic] + offset) * 7 + 9) % 12
    if offset < 0:
        use_sharps = False
    elif offset > 0:
        use_sharps = True
    elif 0 <= tonic_number < 6:
        use_sharps = True
    elif tonic_number > 6:
        use_sharps = False
    notes_sharp = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
    notes_flat = ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B♭', 'B']
    sharp_corrections = [(5, 'E♯'), (0, 'B♯'), (7, 'F𝄪'), (2, 'C𝄪'), (9, 'G𝄪'), (4, 'D𝄪'), (11, 'A𝄪')]
    flat_corrections = [(11, 'C♭'), (4, 'F♭'), (9, 'B𝄫'), (2, 'E𝄫'), (7, 'A𝄫'), (0, 'D𝄫')]
    n_sharps = tonic_number
    if tonic_number == 0 and tonic == 'B':
        n_sharps = 12
    if use_sharps:
        for n in range(0, n_sharps - 6 + 1):
            index, name = sharp_corrections[n]
            notes_sharp[index] = name
        notes = notes_sharp
    else:
        n_flats = (12 - tonic_number) % 12
        for n in range(0, n_flats - 6 + 1):
            index, name = flat_corrections[n]
            notes_flat[index] = name
        notes = notes_flat
    if not unicode:
        translations = str.maketrans({'♯': '#', '𝄪': '##', '♭': 'b', '𝄫': 'bb'})
        notes = list((n.translate(translations) for n in notes))
    return notes