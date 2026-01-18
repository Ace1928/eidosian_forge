from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def is_mouse_press(ev: str) -> bool:
    return 'press' in ev