import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import typing as t
from types import TracebackType
from . import formatting
from . import termui
from . import utils
from ._compat import _find_binary_reader
def make_input_stream(input: t.Optional[t.Union[str, bytes, t.IO[t.Any]]], charset: str) -> t.BinaryIO:
    if hasattr(input, 'read'):
        rv = _find_binary_reader(t.cast(t.IO[t.Any], input))
        if rv is not None:
            return rv
        raise TypeError('Could not find binary reader for input stream.')
    if input is None:
        input = b''
    elif isinstance(input, str):
        input = input.encode(charset)
    return io.BytesIO(input)