import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
@property
def error_string(self):
    """Raw libsndfile error message."""
    if self.code:
        err_str = _snd.sf_error_number(self.code)
        return _ffi.string(err_str).decode('utf-8', 'replace')
    else:
        return '(Garbled error message from libsndfile)'