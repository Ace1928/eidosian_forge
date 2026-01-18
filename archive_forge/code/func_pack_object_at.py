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
def pack_object_at(cursor, offset, as_stream):
    """
    :return: Tuple(abs_data_offset, PackInfo|PackStream)
        an object of the correct type according to the type_id  of the object.
        If as_stream is True, the object will contain a stream, allowing  the
        data to be read decompressed.
    :param data: random accessible data containing all required information
    :parma offset: offset in to the data at which the object information is located
    :param as_stream: if True, a stream object will be returned that can read
        the data, otherwise you receive an info object only"""
    data = cursor.use_region(offset).buffer()
    type_id, uncomp_size, data_rela_offset = pack_object_header_info(data)
    total_rela_offset = None
    delta_info = None
    if type_id == OFS_DELTA:
        i = data_rela_offset
        c = byte_ord(data[i])
        i += 1
        delta_offset = c & 127
        while c & 128:
            c = byte_ord(data[i])
            i += 1
            delta_offset += 1
            delta_offset = (delta_offset << 7) + (c & 127)
        delta_info = delta_offset
        total_rela_offset = i
    elif type_id == REF_DELTA:
        total_rela_offset = data_rela_offset + 20
        delta_info = data[data_rela_offset:total_rela_offset]
    else:
        total_rela_offset = data_rela_offset
    abs_data_offset = offset + total_rela_offset
    if as_stream:
        stream = DecompressMemMapReader(data[total_rela_offset:], False, uncomp_size)
        if delta_info is None:
            return (abs_data_offset, OPackStream(offset, type_id, uncomp_size, stream))
        else:
            return (abs_data_offset, ODeltaPackStream(offset, type_id, uncomp_size, delta_info, stream))
    elif delta_info is None:
        return (abs_data_offset, OPackInfo(offset, type_id, uncomp_size))
    else:
        return (abs_data_offset, ODeltaPackInfo(offset, type_id, uncomp_size, delta_info))