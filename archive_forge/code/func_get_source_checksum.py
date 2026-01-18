import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
def get_source_checksum(self, source: str) -> str:
    """Returns a checksum for the source."""
    return sha1(source.encode('utf-8')).hexdigest()