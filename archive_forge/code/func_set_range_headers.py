from __future__ import annotations
import os
import re
import stat
from typing import NamedTuple
from urllib.parse import quote
import aiofiles
from aiofiles.os import stat as aio_stat
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response, guess_type
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send
def set_range_headers(self, range: ClosedRange) -> None:
    if not self.stat_result:
        raise ValueError('No stat result to set range headers with')
    total_length = self.stat_result.st_size
    content_length = len(range)
    self.headers['content-range'] = f'bytes {range.start}-{range.end}/{total_length}'
    self.headers['content-length'] = str(content_length)
    pass