from contextlib import contextmanager
from .exceptions import ELFParseError, ELFError, DWARFError
from ..construct import ConstructError, ULInt8
import os
def parse_cstring_from_stream(stream, stream_pos=None):
    """ Parse a C-string from the given stream. The string is returned without
        the terminating \x00 byte. If the terminating byte wasn't found, None
        is returned (the stream is exhausted).
        If stream_pos is provided, the stream is seeked to this position before
        the parsing is done. Otherwise, the current position of the stream is
        used.
        Note: a bytes object is returned here, because this is what's read from
        the binary file.
    """
    if stream_pos is not None:
        stream.seek(stream_pos)
    CHUNKSIZE = 64
    chunks = []
    found = False
    while True:
        chunk = stream.read(CHUNKSIZE)
        end_index = chunk.find(b'\x00')
        if end_index >= 0:
            chunks.append(chunk[:end_index])
            found = True
            break
        else:
            chunks.append(chunk)
        if len(chunk) < CHUNKSIZE:
            break
    return b''.join(chunks) if found else None