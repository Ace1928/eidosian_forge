import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
def indexfile_checksum(self):
    """:return: 20 byte sha representing the sha1 hash of this index file"""
    return self._cursor.map()[-20:]