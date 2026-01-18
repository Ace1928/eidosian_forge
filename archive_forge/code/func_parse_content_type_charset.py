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
def parse_content_type_charset(content_type: str) -> str | None:
    msg = email.message.Message()
    msg['content-type'] = content_type
    return msg.get_content_charset(failobj=None)