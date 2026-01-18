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
def skip_lines(self):
    """Internal: skip lines until outer boundary if defined."""
    if not self.outerboundary or self.done:
        return
    next_boundary = b'--' + self.outerboundary
    last_boundary = next_boundary + b'--'
    last_line_lfend = True
    while True:
        line = self.fp.readline(1 << 16)
        self.bytes_read += len(line)
        if not line:
            self.done = -1
            break
        if line.endswith(b'--') and last_line_lfend:
            strippedline = line.strip()
            if strippedline == next_boundary:
                break
            if strippedline == last_boundary:
                self.done = 1
                break
        last_line_lfend = line.endswith(b'\n')