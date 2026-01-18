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
@property
def in_memory(self) -> bool:
    """A boolean representing whether or not this file object is currently
        stored in-memory or on-disk.
        """
    return self._in_memory