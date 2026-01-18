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
def pin_auth(self, request: Request) -> Response:
    """Authenticates with the pin."""
    exhausted = False
    auth = False
    trust = self.check_pin_trust(request.environ)
    pin = t.cast(str, self.pin)
    bad_cookie = False
    if trust is None:
        self._fail_pin_auth()
        bad_cookie = True
    elif trust:
        auth = True
    elif self._failed_pin_auth > 10:
        exhausted = True
    else:
        entered_pin = request.args['pin']
        if entered_pin.strip().replace('-', '') == pin.replace('-', ''):
            self._failed_pin_auth = 0
            auth = True
        else:
            self._fail_pin_auth()
    rv = Response(json.dumps({'auth': auth, 'exhausted': exhausted}), mimetype='application/json')
    if auth:
        rv.set_cookie(self.pin_cookie_name, f'{int(time.time())}|{hash_pin(pin)}', httponly=True, samesite='Strict', secure=request.is_secure)
    elif bad_cookie:
        rv.delete_cookie(self.pin_cookie_name)
    return rv