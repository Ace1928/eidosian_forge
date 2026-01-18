from __future__ import annotations
import codecs
import email.message
import ipaddress
import mimetypes
import os
import re
import time
import typing
from pathlib import Path
from urllib.request import getproxies
import sniffio
from ._types import PrimitiveData
def normalize_header_value(value: str | bytes, encoding: str | None=None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header value.
    """
    if isinstance(value, bytes):
        return value
    return value.encode(encoding or 'ascii')