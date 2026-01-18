from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_additional_interfaces_markdown(interfaces):
    """Renders additional helper classes or types that were extracted earlier."""
    source = ''
    for interface_name, interface in interfaces.items():
        source += f'\n## `{interface_name}`\n```python\n{interface['source']}\n```\n'
    return source