from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def dict_repr(self, d: dict[int, None] | dict[str, int] | dict[str | int, int], recursive: bool, limit: int=5) -> str:
    if recursive:
        return _add_subclass_info('{...}', d, dict)
    buf = ['{']
    have_extended_section = False
    for idx, (key, value) in enumerate(d.items()):
        if idx:
            buf.append(', ')
        if idx == limit - 1:
            buf.append('<span class="extended">')
            have_extended_section = True
        buf.append(f'<span class="pair"><span class="key">{self.repr(key)}</span>: <span class="value">{self.repr(value)}</span></span>')
    if have_extended_section:
        buf.append('</span>')
    buf.append('}')
    return _add_subclass_info(''.join(buf), d, dict)