import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def get_text_stdout(encoding: t.Optional[str]=None, errors: t.Optional[str]=None) -> t.TextIO:
    rv = _get_windows_console_stream(sys.stdout, encoding, errors)
    if rv is not None:
        return rv
    return _force_correct_text_writer(sys.stdout, encoding, errors, force_writable=True)