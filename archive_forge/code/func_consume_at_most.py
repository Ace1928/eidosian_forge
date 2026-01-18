import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def consume_at_most(self, nbytes: int) -> bytes:
    if not nbytes:
        return bytearray()
    data = self.buffer[self.bytes_used:self.bytes_used + nbytes]
    self.bytes_used += len(data)
    return data