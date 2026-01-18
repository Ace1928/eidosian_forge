from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def regex_repr(self, obj: t.Pattern) -> str:
    pattern = repr(obj.pattern)
    pattern = codecs.decode(pattern, 'unicode-escape', 'ignore')
    pattern = f'r{pattern}'
    return f're.compile(<span class="string regex">{pattern}</span>)'