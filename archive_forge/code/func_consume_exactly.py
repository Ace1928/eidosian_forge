import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def consume_exactly(self, nbytes: int) -> Optional[bytes]:
    if len(self.buffer) - self.bytes_used < nbytes:
        return None
    return self.consume_at_most(nbytes)