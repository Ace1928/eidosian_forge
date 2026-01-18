from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
def flush_to_disk(self) -> None:
    """If the file is already on-disk, do nothing.  Otherwise, copy from
        the in-memory buffer to a disk file, and then reassign our internal
        file object to this new disk file.

        Note that if you attempt to flush a file that is already on-disk, a
        warning will be logged to this module's logger.
        """
    if not self._in_memory:
        self.logger.warning("Trying to flush to disk when we're not in memory")
        return
    self._fileobj.seek(0)
    new_file = self._get_disk_file()
    shutil.copyfileobj(self._fileobj, new_file)
    new_file.seek(self._bytes_written)
    old_fileobj = self._fileobj
    self._fileobj = new_file
    self._in_memory = False
    old_fileobj.close()