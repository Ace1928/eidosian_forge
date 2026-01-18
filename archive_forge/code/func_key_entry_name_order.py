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
def key_entry_name_order(entry):
    """Sort key for tree entry in name order."""
    return entry[0]