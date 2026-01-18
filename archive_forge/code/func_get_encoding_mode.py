from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def get_encoding_mode() -> Literal['wide', 'narrow', 'utf8']:
    """
    Get the mode Urwid is using when processing text strings.
    Returns 'narrow' for 8-bit encodings, 'wide' for CJK encodings
    or 'utf8' for UTF-8 encodings.
    """
    return str_util.get_byte_encoding()