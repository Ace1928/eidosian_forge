import os
import re
import zlib
from typing import IO, TYPE_CHECKING, Callable, Iterator
from sphinx.util import logging
from sphinx.util.typing import Inventory
def read_compressed_lines(self) -> Iterator[str]:
    buf = b''
    for chunk in self.read_compressed_chunks():
        buf += chunk
        pos = buf.find(b'\n')
        while pos != -1:
            yield buf[:pos].decode()
            buf = buf[pos + 1:]
            pos = buf.find(b'\n')