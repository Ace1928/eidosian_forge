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
def format_mime(obj):
    """
    Formats object using _repr_x_ methods.
    """
    if isinstance(obj, str):
        return (escape(obj), 'text/plain')
    mimebundle = eval_formatter(obj, '_repr_mimebundle_')
    if isinstance(mimebundle, tuple):
        format_dict, _ = mimebundle
    else:
        format_dict = mimebundle
    output, not_available = (None, [])
    for method, mime_type in reversed(list(MIME_METHODS.items())):
        if mime_type in format_dict:
            output = format_dict[mime_type]
        elif isinstance(obj, type) and method != '__repr__':
            output = None
        else:
            output = eval_formatter(obj, method)
        if output is None:
            continue
        elif mime_type not in MIME_RENDERERS:
            not_available.append(mime_type)
            continue
        break
    if output is None:
        output = repr(output)
        mime_type = 'text/plain'
    elif isinstance(output, tuple):
        output, meta = output
    else:
        meta = {}
    content, mime_type = MIME_RENDERERS[mime_type](output, meta, mime_type)
    if mime_type == 'text/plain':
        content = escape(content)
    return (content, mime_type)