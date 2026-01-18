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
def lookup_path(self, lookup_obj, path):
    """Look up an object in a Git tree.

        Args:
          lookup_obj: Callback for retrieving object by SHA1
          path: Path to lookup
        Returns: A tuple of (mode, SHA) of the resulting path.
        """
    parts = path.split(b'/')
    sha = self.id
    mode = None
    for i, p in enumerate(parts):
        if not p:
            continue
        if mode is not None and S_ISGITLINK(mode):
            raise SubmoduleEncountered(b'/'.join(parts[:i]), sha)
        obj = lookup_obj(sha)
        if not isinstance(obj, Tree):
            raise NotTreeError(sha)
        mode, sha = obj[p]
    return (mode, sha)