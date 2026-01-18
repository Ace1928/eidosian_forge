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
def port_or_default(url: URL) -> int | None:
    if url.port is not None:
        return url.port
    return {'http': 80, 'https': 443}.get(url.scheme)