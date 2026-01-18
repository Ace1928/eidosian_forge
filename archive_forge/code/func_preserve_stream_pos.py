from contextlib import contextmanager
from .exceptions import ELFParseError, ELFError, DWARFError
from ..construct import ConstructError, ULInt8
import os
@contextmanager
def preserve_stream_pos(stream):
    """ Usage:
        # stream has some position FOO (return value of stream.tell())
        with preserve_stream_pos(stream):
            # do stuff that manipulates the stream
        # stream still has position FOO
    """
    saved_pos = stream.tell()
    yield
    stream.seek(saved_pos)