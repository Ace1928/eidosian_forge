from __future__ import annotations
from typing import Generator
def read_exact(self, n: int) -> Generator[None, None, bytes]:
    """
        Read a given number of bytes from the stream.

        This is a generator-based coroutine.

        Args:
            n: how many bytes to read.

        Raises:
            EOFError: if the stream ends in less than ``n`` bytes.

        """
    assert n >= 0
    while len(self.buffer) < n:
        if self.eof:
            p = len(self.buffer)
            raise EOFError(f'stream ends after {p} bytes, expected {n} bytes')
        yield
    r = self.buffer[:n]
    del self.buffer[:n]
    return r