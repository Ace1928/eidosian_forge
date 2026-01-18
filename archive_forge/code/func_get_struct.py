import contextlib
import struct
from typing import Iterator, Optional, Tuple
import dns.exception
import dns.name
def get_struct(self, format: str) -> Tuple:
    return struct.unpack(format, self.get_bytes(struct.calcsize(format)))