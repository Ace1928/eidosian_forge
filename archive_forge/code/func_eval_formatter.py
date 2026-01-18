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
def eval_formatter(obj, print_method):
    """
    Evaluates a formatter method.
    """
    if print_method == '__repr__':
        return repr(obj)
    elif hasattr(obj, print_method):
        if print_method == 'savefig':
            buf = io.BytesIO()
            obj.savefig(buf, format='png')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        return getattr(obj, print_method)()
    elif print_method == '_repr_mimebundle_':
        return ({}, {})
    return None