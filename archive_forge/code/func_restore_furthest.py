import contextlib
import struct
from typing import Iterator, Optional, Tuple
import dns.exception
import dns.name
@contextlib.contextmanager
def restore_furthest(self) -> Iterator:
    try:
        yield None
    finally:
        self.current = self.furthest