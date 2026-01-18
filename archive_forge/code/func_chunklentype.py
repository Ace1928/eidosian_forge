from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def chunklentype(self):
    """Reads just enough of the input to determine the next
        chunk's length and type, returned as a (*length*, *type*) pair
        where *type* is a string.  If there are no more chunks, ``None``
        is returned.
        """
    x = self.file.read(8)
    if not x:
        return None
    if len(x) != 8:
        raise FormatError('End of file whilst reading chunk length and type.')
    length, type = struct.unpack('!I4s', x)
    type = bytestostr(type)
    if length > 2 ** 31 - 1:
        raise FormatError('Chunk %s is too large: %d.' % (type, length))
    return (length, type)