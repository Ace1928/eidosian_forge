import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def get_text_stderr(encoding: t.Optional[str]=None, errors: t.Optional[str]=None) -> t.TextIO:
    rv = _get_windows_console_stream(sys.stderr, encoding, errors)
    if rv is not None:
        return rv
    return _force_correct_text_writer(sys.stderr, encoding, errors, force_writable=True)