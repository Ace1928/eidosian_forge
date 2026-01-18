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
def render_debugger_html(self, evalex: bool, secret: str, evalex_trusted: bool) -> str:
    exc_lines = list(self._te.format_exception_only())
    plaintext = ''.join(self._te.format())
    return PAGE_HTML % {'evalex': 'true' if evalex else 'false', 'evalex_trusted': 'true' if evalex_trusted else 'false', 'console': 'false', 'title': escape(exc_lines[0]), 'exception': escape(''.join(exc_lines)), 'exception_type': escape(self._te.exc_type.__name__), 'summary': self.render_traceback_html(include_title=False), 'plaintext': escape(plaintext), 'plaintext_cs': re.sub('-{2,}', '-', plaintext), 'secret': secret}