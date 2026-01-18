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
def display_console(self, request: Request) -> Response:
    """Display a standalone shell."""
    if 0 not in self.frames:
        if self.console_init_func is None:
            ns = {}
        else:
            ns = dict(self.console_init_func())
        ns.setdefault('app', self.app)
        self.frames[0] = _ConsoleFrame(ns)
    is_trusted = bool(self.check_pin_trust(request.environ))
    return Response(render_console_html(secret=self.secret, evalex_trusted=is_trusted), mimetype='text/html')