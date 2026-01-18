from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def fallback_repr(self) -> str:
    try:
        info = ''.join(format_exception_only(*sys.exc_info()[:2]))
    except Exception:
        info = '?'
    return f'<span class="brokenrepr">&lt;broken repr ({escape(info.strip())})&gt;</span>'