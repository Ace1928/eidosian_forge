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
def serializable_property(name: str, docstring: Optional[str]=None):
    """A property that helps tracking whether serialization is necessary."""

    def set(obj, value):
        setattr(obj, '_' + name, value)
        obj._needs_serialization = True

    def get(obj):
        return getattr(obj, '_' + name)
    return property(get, set, doc=docstring)