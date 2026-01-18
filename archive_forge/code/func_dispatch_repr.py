from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def dispatch_repr(self, obj: t.Any, recursive: bool) -> str:
    if obj is helper:
        return f'<span class="help">{helper!r}</span>'
    if isinstance(obj, (int, float, complex)):
        return f'<span class="number">{obj!r}</span>'
    if isinstance(obj, str) or isinstance(obj, bytes):
        return self.string_repr(obj)
    if isinstance(obj, RegexType):
        return self.regex_repr(obj)
    if isinstance(obj, list):
        return self.list_repr(obj, recursive)
    if isinstance(obj, tuple):
        return self.tuple_repr(obj, recursive)
    if isinstance(obj, set):
        return self.set_repr(obj, recursive)
    if isinstance(obj, frozenset):
        return self.frozenset_repr(obj, recursive)
    if isinstance(obj, dict):
        return self.dict_repr(obj, recursive)
    if isinstance(obj, deque):
        return self.deque_repr(obj, recursive)
    return self.object_repr(obj)