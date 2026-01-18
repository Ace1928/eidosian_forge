from __future__ import annotations
import importlib.util
import mimetypes
import os
import posixpath
import typing as t
from datetime import datetime
from datetime import timezone
from io import BytesIO
from time import time
from zlib import adler32
from ..http import http_date
from ..http import is_resource_modified
from ..security import safe_join
from ..utils import get_content_type
from ..wsgi import get_path_info
from ..wsgi import wrap_file
def generate_etag(self, mtime: datetime, file_size: int, real_filename: str) -> str:
    real_filename = os.fsencode(real_filename)
    timestamp = mtime.timestamp()
    checksum = adler32(real_filename) & 4294967295
    return f'wzsdm-{timestamp}-{file_size}-{checksum}'