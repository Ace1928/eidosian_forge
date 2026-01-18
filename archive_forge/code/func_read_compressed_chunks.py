import os
import re
import zlib
from typing import IO, TYPE_CHECKING, Callable, Iterator
from sphinx.util import logging
from sphinx.util.typing import Inventory
def read_compressed_chunks(self) -> Iterator[bytes]:
    decompressor = zlib.decompressobj()
    while not self.eof:
        self.read_buffer()
        yield decompressor.decompress(self.buffer)
        self.buffer = b''
    yield decompressor.flush()