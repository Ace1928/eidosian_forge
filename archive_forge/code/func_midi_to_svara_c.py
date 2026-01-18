from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
@vectorize(excluded=['Sa', 'mela', 'abbr', 'octave', 'unicode'])
def midi_to_svara_c(midi: Union[float, np.ndarray], *, Sa: _FloatLike_co, mela: Union[int, str], abbr: bool=True, octave: bool=True, unicode: bool=True) -> Union[str, np.ndarray]:
    """Convert MIDI numbers to Carnatic svara within a given melakarta raga

    Parameters
    ----------
    midi : numeric
        The MIDI numbers to convert

    Sa : number > 0
        MIDI number of the reference Sa.

        Default: 60 (261.6 Hz, `C4`)

    mela : int or str
        The name or index of the melakarta raga

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'R1', 'R2', 'G1', 'G2', ...)

        If `False`, return long-form names ('Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information and subscript
        numbers.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

    Returns
    -------
    svara : str or np.ndarray of str
        The svara corresponding to the given MIDI number(s)

    See Also
    --------
    hz_to_svara_c
    note_to_svara_c
    mela_to_degrees
    mela_to_svara
    list_mela
    """
    svara_num = int(np.round(midi - Sa))
    svara_map = notation.mela_to_svara(mela, abbr=abbr, unicode=unicode)
    svara = svara_map[svara_num % 12]
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