import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def check_user_identity(identity):
    """Verify that a user identity is formatted correctly.

    Args:
      identity: User identity bytestring
    Raises:
      InvalidUserIdentity: Raised when identity is invalid
    """
    try:
        fst, snd = identity.split(b' <', 1)
    except ValueError as exc:
        raise InvalidUserIdentity(identity) from exc
    if b'>' not in snd:
        raise InvalidUserIdentity(identity)
    if b'\x00' in identity or b'\n' in identity:
        raise InvalidUserIdentity(identity)