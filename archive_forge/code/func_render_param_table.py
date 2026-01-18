from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_param_table(params):
    """Renders the parameter table for the package."""
    table = '<table>\n<thead>\n<tr>\n<th align="left">name</th>\n<th align="left" style="width: 25%;">type</th>\n<th align="left">default</th>\n<th align="left">description</th>\n</tr>\n</thead>\n<tbody>'
    for param_name, param in params.items():
        table += f'\n<tr>\n<td align="left"><code>{param_name}</code></td>\n<td align="left" style="width: 25%;">\n\n```python\n{param['type']}\n```\n\n</td>\n<td align="left"><code>{param['default']}</code></td>\n<td align="left">{param['description']}</td>\n</tr>\n'
    return table + '</tbody></table>'