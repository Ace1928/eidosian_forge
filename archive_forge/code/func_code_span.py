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
def code_span(s, fg, bg, cursor=-1) -> str:
    code_fg = _code_colours[fg]
    code_bg = _code_colours[bg]
    if cursor >= 0:
        c_off, _ign = calc_text_pos(s, 0, len(s), cursor)
        c2_off = move_next_char(s, c_off, len(s))
        return code_fg + code_bg + s[:c_off] + '\n' + code_bg + code_fg + s[c_off:c2_off] + '\n' + code_fg + code_bg + s[c2_off:] + '\n'
    return f'{code_fg + code_bg + s}\n'