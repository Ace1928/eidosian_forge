from __future__ import annotations
import dataclasses
import glob
import html
import os
import pathlib
import random
import selectors
import signal
import socket
import string
import sys
import tempfile
import typing
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width, move_next_char
from urwid.util import StoppingContext, get_encoding
from .common import BaseScreen
def handle_short_request() -> bool:
    """
    Handle short requests such as passing keystrokes to the application
    or sending the initial html page.  If returns True, then this
    function recognised and handled a short request, and the calling
    script should immediately exit.

    web_display.set_preferences(..) should be called before calling this
    function for the preferences to take effect
    """
    if not is_web_request():
        return False
    if os.environ['REQUEST_METHOD'] == 'GET':
        sys.stdout.write('Content-type: text/html\r\n\r\n' + html.escape(_prefs.app_name).join(_html_page))
        return True
    if os.environ['REQUEST_METHOD'] != 'POST':
        return False
    if 'HTTP_X_URWID_ID' not in os.environ:
        return False
    urwid_id = os.environ['HTTP_X_URWID_ID']
    if len(urwid_id) > 20:
        sys.stdout.write('Status: 414 URI Too Long\r\n\r\n')
        return True
    for c in urwid_id:
        if c not in string.digits:
            sys.stdout.write('Status: 403 Forbidden\r\n\r\n')
            return True
    if os.environ.get('HTTP_X_URWID_METHOD', None) == 'polling':
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(os.path.join(_prefs.pipe_dir, f'urwid{urwid_id}.update'))
            data = f'Content-type: text/plain\r\n\r\n{s.recv(BUF_SZ)}'
            while data:
                sys.stdout.write(data)
                data = s.recv(BUF_SZ)
        except OSError:
            sys.stdout.write('Status: 404 Not Found\r\n\r\n')
            return True
        return True
    try:
        fd = os.open(os.path.join(_prefs.pipe_dir, f'urwid{urwid_id}.in'), os.O_WRONLY)
    except OSError:
        sys.stdout.write('Status: 404 Not Found\r\n\r\n')
        return True
    keydata = sys.stdin.read(MAX_READ)
    os.write(fd, keydata.encode('ascii'))
    os.close(fd)
    sys.stdout.write('Content-type: text/plain\r\n\r\n')
    return True