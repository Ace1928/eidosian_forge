from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
def readuntil(self, char=b'\n', blocks=None):
    """Return data between current position and first occurrence of char

        char is included in the output, except if the end of the tile is
        encountered first.

        Parameters
        ----------
        char: bytes
            Thing to find
        blocks: None or int
            How much to read in each go. Defaults to file blocksize - which may
            mean a new read on every call.
        """
    out = []
    while True:
        start = self.tell()
        part = self.read(blocks or self.blocksize)
        if len(part) == 0:
            break
        found = part.find(char)
        if found > -1:
            out.append(part[:found + len(char)])
            self.seek(start + found + len(char))
            break
        out.append(part)
    return b''.join(out)