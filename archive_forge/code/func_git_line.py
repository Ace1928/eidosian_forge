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
def git_line(*items):
    """Formats items into a space separated line."""
    return b' '.join(items) + b'\n'