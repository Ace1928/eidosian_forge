import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def splitlines(self) -> List[bytes]:
    """Return list of lines in this blob.

        This preserves the original line endings.
        """
    chunks = self.chunked
    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks[0].splitlines(True)
    remaining = None
    ret = []
    for chunk in chunks:
        lines = chunk.splitlines(True)
        if len(lines) > 1:
            ret.append((remaining or b'') + lines[0])
            ret.extend(lines[1:-1])
            remaining = lines[-1]
        elif len(lines) == 1:
            if remaining is None:
                remaining = lines.pop()
            else:
                remaining += lines.pop()
    if remaining is not None:
        ret.append(remaining)
    return ret