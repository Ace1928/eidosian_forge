import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def read_previous_line(stream: StreamType) -> bytes:
    """
    Given a byte stream with current position X, return the previous line.

    All characters between the first CR/LF byte found before X
    (or, the start of the file, if no such byte is found) and position X
    After this call, the stream will be positioned one byte after the
    first non-CRLF character found beyond the first CR/LF byte before X,
    or, if no such byte is found, at the beginning of the stream.

    Args:
        stream: StreamType:

    Returns:
        The data which was read.
    """
    line_content = []
    found_crlf = False
    if stream.tell() == 0:
        raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
    while True:
        to_read = min(DEFAULT_BUFFER_SIZE, stream.tell())
        if to_read == 0:
            break
        block = read_block_backwards(stream, to_read)
        idx = len(block) - 1
        if not found_crlf:
            while idx >= 0 and block[idx] not in b'\r\n':
                idx -= 1
            if idx >= 0:
                found_crlf = True
        if found_crlf:
            line_content.append(block[idx + 1:])
            while idx >= 0 and block[idx] in b'\r\n':
                idx -= 1
        else:
            line_content.append(block)
        if idx >= 0:
            stream.seek(idx + 1, SEEK_CUR)
            break
    return b''.join(line_content[::-1])