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
def normalize_header_key(value: str | bytes, lower: bool, encoding: str | None=None) -> bytes:
    """
    Coerce str/bytes into a strictly byte-wise HTTP header key.
    """
    if isinstance(value, bytes):
        bytes_value = value
    else:
        bytes_value = value.encode(encoding or 'ascii')
    return bytes_value.lower() if lower else bytes_value