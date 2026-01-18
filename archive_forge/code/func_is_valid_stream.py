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
def is_valid_stream(self, sha, use_crc=False):
    """
        Verify that the stream at the given sha is valid.

        :param use_crc: if True, the index' crc is run over the compressed stream of
            the object, which is much faster than checking the sha1. It is also
            more prone to unnoticed corruption or manipulation.
        :param sha: 20 byte sha1 of the object whose stream to verify
            whether the compressed stream of the object is valid. If it is
            a delta, this only verifies that the delta's data is valid, not the
            data of the actual undeltified object, as it depends on more than
            just this stream.
            If False, the object will be decompressed and the sha generated. It must
            match the given sha

        :return: True if the stream is valid
        :raise UnsupportedOperation: If the index is version 1 only
        :raise BadObject: sha was not found"""
    if use_crc:
        if self._index.version() < 2:
            raise UnsupportedOperation("Version 1 indices do not contain crc's, verify by sha instead")
        index = self._sha_to_index(sha)
        offset = self._index.offset(index)
        next_offset = self._offset_map[offset]
        crc_value = self._index.crc(index)
        crc_update = zlib.crc32
        pack_data = self._pack.data()
        cur_pos = offset
        this_crc_value = 0
        while cur_pos < next_offset:
            rbound = min(cur_pos + chunk_size, next_offset)
            size = rbound - cur_pos
            this_crc_value = crc_update(pack_data[cur_pos:cur_pos + size], this_crc_value)
            cur_pos += size
        return this_crc_value & 4294967295 == crc_value
    else:
        shawriter = Sha1Writer()
        stream = self._object(sha, as_stream=True)
        write_object(stream.type, stream.size, stream.read, shawriter.write)
        assert shawriter.sha(as_hex=False) == sha
        return shawriter.sha(as_hex=False) == sha
    return True