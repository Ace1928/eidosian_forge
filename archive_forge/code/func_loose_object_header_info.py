import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def loose_object_header_info(m):
    """
    :return: tuple(type_string, uncompressed_size_in_bytes) the type string of the
        object as well as its uncompressed size in bytes.
    :param m: memory map from which to read the compressed object data"""
    decompress_size = 8192
    hdr = decompressobj().decompress(m, decompress_size)
    type_name, size = hdr[:hdr.find(NULL_BYTE)].split(BYTE_SPACE)
    return (type_name, int(size))