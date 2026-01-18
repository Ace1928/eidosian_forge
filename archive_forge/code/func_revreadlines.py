import logging
import re
from typing import (
from . import settings
from .utils import choplist
def revreadlines(self) -> Iterator[bytes]:
    """Fetches a next line backword.

        This is used to locate the trailers at the end of a file.
        """
    self.fp.seek(0, 2)
    pos = self.fp.tell()
    buf = b''
    while 0 < pos:
        prevpos = pos
        pos = max(0, pos - self.BUFSIZ)
        self.fp.seek(pos)
        s = self.fp.read(prevpos - pos)
        if not s:
            break
        while 1:
            n = max(s.rfind(b'\r'), s.rfind(b'\n'))
            if n == -1:
                buf = s + buf
                break
            yield (s[n:] + buf)
            s = s[:n]
            buf = b''
    return