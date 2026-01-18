import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def default_subtype(format):
    """Return the default subtype for a given format.

    Examples
    --------
    >>> import soundfile as sf
    >>> sf.default_subtype('WAV')
    'PCM_16'
    >>> sf.default_subtype('MAT5')
    'DOUBLE'

    """
    _check_format(format)
    return _default_subtypes.get(format.upper())