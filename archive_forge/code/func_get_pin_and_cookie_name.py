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
def get_pin_and_cookie_name(app: WSGIApplication) -> tuple[str, str] | tuple[None, None]:
    """Given an application object this returns a semi-stable 9 digit pin
    code and a random key.  The hope is that this is stable between
    restarts to not make debugging particularly frustrating.  If the pin
    was forcefully disabled this returns `None`.

    Second item in the resulting tuple is the cookie name for remembering.
    """
    pin = os.environ.get('WERKZEUG_DEBUG_PIN')
    rv = None
    num = None
    if pin == 'off':
        return (None, None)
    if pin is not None and pin.replace('-', '').isdecimal():
        if '-' in pin:
            rv = pin
        else:
            num = pin
    modname = getattr(app, '__module__', t.cast(object, app).__class__.__module__)
    username: str | None
    try:
        username = getpass.getuser()
    except (ImportError, KeyError):
        username = None
    mod = sys.modules.get(modname)
    probably_public_bits = [username, modname, getattr(app, '__name__', type(app).__name__), getattr(mod, '__file__', None)]
    private_bits = [str(uuid.getnode()), get_machine_id()]
    h = hashlib.sha1()
    for bit in chain(probably_public_bits, private_bits):
        if not bit:
            continue
        if isinstance(bit, str):
            bit = bit.encode('utf-8')
        h.update(bit)
    h.update(b'cookiesalt')
    cookie_name = f'__wzd{h.hexdigest()[:20]}'
    if num is None:
        h.update(b'pinsalt')
        num = f'{int(h.hexdigest(), 16):09d}'[:9]
    if rv is None:
        for group_size in (5, 4, 3):
            if len(num) % group_size == 0:
                rv = '-'.join((num[x:x + group_size].rjust(group_size, '0') for x in range(0, len(num), group_size)))
                break
        else:
            rv = num
    return (rv, cookie_name)