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
def collect_streams_at_offset(self, offset):
    """
        As the version in the PackFile, but can resolve REF deltas within this pack
        For more info, see ``collect_streams``

        :param offset: offset into the pack file at which the object can be found"""
    streams = self._pack.collect_streams(offset)
    if streams[-1].type_id == REF_DELTA:
        stream = streams[-1]
        while stream.type_id in delta_types:
            if stream.type_id == REF_DELTA:
                if isinstance(stream.delta_info, memoryview):
                    sindex = self._index.sha_to_index(stream.delta_info.tobytes())
                else:
                    sindex = self._index.sha_to_index(stream.delta_info)
                if sindex is None:
                    break
                stream = self._pack.stream(self._index.offset(sindex))
                streams.append(stream)
            else:
                stream = self._pack.stream(stream.delta_info)
                streams.append(stream)
    return streams