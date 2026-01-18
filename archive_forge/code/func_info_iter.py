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
def info_iter(self):
    """
        :return: Iterator over all objects in this pack. The iterator yields
            OInfo instances"""
    return self._iter_objects(as_stream=False)