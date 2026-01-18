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
def collect_streams(self, sha):
    """
        As ``PackFile.collect_streams``, but takes a sha instead of an offset.
        Additionally, ref_delta streams will be resolved within this pack.
        If this is not possible, the stream will be left alone, hence it is adivsed
        to check for unresolved ref-deltas and resolve them before attempting to
        construct a delta stream.

        :param sha: 20 byte sha1 specifying the object whose related streams you want to collect
        :return: list of streams, first being the actual object delta, the last being
            a possibly unresolved base object.
        :raise BadObject:"""
    return self.collect_streams_at_offset(self._index.offset(self._sha_to_index(sha)))