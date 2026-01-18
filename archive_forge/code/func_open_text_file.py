from __future__ import annotations
import io
import json
import os
import typing as t
from .encoding import (
def open_text_file(path: str, mode: str='r') -> t.IO[str]:
    """Open the given path for text access."""
    if 'b' in mode:
        raise Exception('mode cannot include "b" for text files: %s' % mode)
    return io.open(to_bytes(path), mode, encoding=ENCODING)