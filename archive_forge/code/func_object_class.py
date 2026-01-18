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
def object_class(type: Union[bytes, int]) -> Optional[Type['ShaFile']]:
    """Get the object class corresponding to the given type.

    Args:
      type: Either a type name string or a numeric type.
    Returns: The ShaFile subclass corresponding to the given type, or None if
        type is not a valid type name/number.
    """
    return _TYPE_MAP.get(type, None)