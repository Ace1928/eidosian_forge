from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def render_object_dump(self, items: list[tuple[str, str]], title: str, repr: str | None=None) -> str:
    html_items = []
    for key, value in items:
        html_items.append(f'<tr><th>{escape(key)}<td><pre class=repr>{value}</pre>')
    if not html_items:
        html_items.append('<tr><td><em>Nothing</em>')
    return OBJECT_DUMP_HTML % {'title': escape(title), 'repr': f'<pre class=repr>{(repr if repr else '')}</pre>', 'items': '\n'.join(html_items)}