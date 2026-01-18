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
def parse_options_header(value: str | bytes) -> tuple[bytes, dict[bytes, bytes]]:
    """
    Parses a Content-Type header into a value in the following format:
        (content_type, {parameters})
    """
    if not value:
        return (b'', {})
    if isinstance(value, bytes):
        value = value.decode('latin-1')
    assert isinstance(value, str), 'Value should be a string by now'
    if ';' not in value:
        return (value.lower().strip().encode('latin-1'), {})
    message = Message()
    message['content-type'] = value
    params = message.get_params()
    assert params, 'At least the content type value should be present'
    ctype = params.pop(0)[0].encode('latin-1')
    options = {}
    for param in params:
        key, value = param
        if isinstance(value, tuple):
            value = value[-1]
        if key == 'filename':
            if value[1:3] == ':\\' or value[:2] == '\\\\':
                value = value.split('\\')[-1]
        options[key.encode('latin-1')] = value.encode('latin-1')
    return (ctype, options)