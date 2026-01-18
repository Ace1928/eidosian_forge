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
def same_origin(url: URL, other: URL) -> bool:
    """
    Return 'True' if the given URLs share the same origin.
    """
    return url.scheme == other.scheme and url.host == other.host and (port_or_default(url) == port_or_default(other))