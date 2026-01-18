from __future__ import annotations
import getpass
import hashlib
import json
import os
import pkgutil
import re
import sys
import time
import typing as t
import uuid
from contextlib import ExitStack
from io import BytesIO
from itertools import chain
from os.path import basename
from os.path import join
from zlib import adler32
from .._internal import _log
from ..exceptions import NotFound
from ..http import parse_cookie
from ..security import gen_salt
from ..utils import send_file
from ..wrappers.request import Request
from ..wrappers.response import Response
from .console import Console
from .tbtools import DebugFrameSummary
from .tbtools import DebugTraceback
from .tbtools import render_console_html
def check_pin_trust(self, environ: WSGIEnvironment) -> bool | None:
    """Checks if the request passed the pin test.  This returns `True` if the
        request is trusted on a pin/cookie basis and returns `False` if not.
        Additionally if the cookie's stored pin hash is wrong it will return
        `None` so that appropriate action can be taken.
        """
    if self.pin is None:
        return True
    val = parse_cookie(environ).get(self.pin_cookie_name)
    if not val or '|' not in val:
        return False
    ts_str, pin_hash = val.split('|', 1)
    try:
        ts = int(ts_str)
    except ValueError:
        return False
    if pin_hash != hash_pin(self.pin):
        return None
    return time.time() - PIN_TIME < ts