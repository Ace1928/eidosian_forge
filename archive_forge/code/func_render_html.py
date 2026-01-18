from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
def render_html(self, mark_library: bool) -> str:
    context = 5
    lines = linecache.getlines(self.filename)
    line_idx = self.lineno - 1
    start_idx = max(0, line_idx - context)
    stop_idx = min(len(lines), line_idx + context + 1)
    rendered_lines = []

    def render_line(line: str, cls: str) -> None:
        line = line.expandtabs().rstrip()
        stripped_line = line.strip()
        prefix = len(line) - len(stripped_line)
        colno = getattr(self, 'colno', 0)
        end_colno = getattr(self, 'end_colno', 0)
        if cls == 'current' and colno and end_colno:
            arrow = f'\n<span class="ws">{' ' * prefix}</span>{' ' * (colno - prefix)}{'^' * (end_colno - colno)}'
        else:
            arrow = ''
        rendered_lines.append(f'<pre class="line {cls}"><span class="ws">{' ' * prefix}</span>{(escape(stripped_line) if stripped_line else ' ')}{(arrow if arrow else '')}</pre>')
    if lines:
        for line in lines[start_idx:line_idx]:
            render_line(line, 'before')
        render_line(lines[line_idx], 'current')
        for line in lines[line_idx + 1:stop_idx]:
            render_line(line, 'after')
    return FRAME_HTML % {'id': id(self), 'filename': escape(self.filename), 'lineno': self.lineno, 'function_name': escape(self.name), 'lines': '\n'.join(rendered_lines), 'library': 'library' if mark_library and self.is_library else ''}