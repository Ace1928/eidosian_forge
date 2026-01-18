import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def read_exactly(stream, size):
    """Read exactly size bytes from stream.

    :param stream: A file like object to read bytes from. Must support
        read(<count>) and return bytes.
    :param size: The number of bytes to retrieve.
    """
    data = b''
    remaining = size
    while remaining:
        read = stream.read(remaining)
        if len(read) == 0:
            raise ParseError('Short read - got %d bytes, wanted %d bytes' % (len(data), size))
        data += read
        remaining -= len(read)
    return data