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
def set_raw_chunks(self, chunks: List[bytes], sha: Optional[ObjectID]=None) -> None:
    """Set the contents of this object from a list of chunks."""
    self._chunked_text = chunks
    self._deserialize(chunks)
    if sha is None:
        self._sha = None
    else:
        self._sha = FixedSha(sha)
    self._needs_serialization = False