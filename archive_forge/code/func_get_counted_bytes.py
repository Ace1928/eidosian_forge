import contextlib
import struct
from typing import Iterator, Optional, Tuple
import dns.exception
import dns.name
def get_counted_bytes(self, length_size: int=1) -> bytes:
    length = int.from_bytes(self.get_bytes(length_size), 'big')
    return self.get_bytes(length)