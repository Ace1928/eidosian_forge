import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
def cache_forever_headers(now=None):
    if now is None:
        now = time.time()
    return [('Date', date_time_string(now)), ('Expires', date_time_string(now + 31536000)), ('Cache-Control', 'public, max-age=31536000')]