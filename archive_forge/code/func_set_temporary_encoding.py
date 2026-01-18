from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
@contextlib.contextmanager
def set_temporary_encoding(encoding_name: str) -> Generator[None, None, None]:
    """Internal helper for encoding specific validation in unittests/doctests.

    Not exported globally.
    """
    old_encoding = _target_encoding
    try:
        set_encoding(encoding_name)
        yield
    finally:
        set_encoding(old_encoding)