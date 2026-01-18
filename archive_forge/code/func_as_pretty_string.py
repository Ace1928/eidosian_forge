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
def as_pretty_string(self):
    text: List[str] = []
    for name, mode, hexsha in self.iteritems():
        text.append(pretty_format_tree_entry(name, mode, hexsha))
    return ''.join(text)