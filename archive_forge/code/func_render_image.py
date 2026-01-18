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
def render_image(value, meta, mime):
    data = f'data:{mime};charset=utf-8;base64,{value}'
    attrs = ' '.join(['{k}="{v}"' for k, v in meta.items()])
    return (f'<img src="{data}" {attrs}</img>', 'text/html')