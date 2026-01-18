from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
def read_lines_to_eof(self):
    """Internal: read lines until EOF."""
    while 1:
        line = self.fp.readline(1 << 16)
        self.bytes_read += len(line)
        if not line:
            self.done = -1
            break
        self.__write(line)