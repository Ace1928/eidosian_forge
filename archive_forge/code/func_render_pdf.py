from __future__ import annotations
import ast
import base64
import copy
import io
import pathlib
import pkgutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from html import escape
from textwrap import dedent
from typing import Any, Dict, List
import markdown
def render_pdf(value, meta, mime):
    data = value.encode('utf-8')
    base64_pdf = base64.b64encode(data).decode('utf-8')
    src = f'data:application/pdf;base64,{base64_pdf}'
    return (f'<embed src="{src}" width="100%" height="100%" type="application/pdf">', 'text/html')