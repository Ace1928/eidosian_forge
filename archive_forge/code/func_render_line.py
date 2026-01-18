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