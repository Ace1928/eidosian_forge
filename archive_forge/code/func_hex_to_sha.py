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
def hex_to_sha(hex):
    """Takes a hex sha and returns a binary sha."""
    assert len(hex) == 40, 'Incorrect length of hexsha: %s' % hex
    try:
        return binascii.unhexlify(hex)
    except TypeError as exc:
        if not isinstance(hex, bytes):
            raise
        raise ValueError(exc.args[0]) from exc