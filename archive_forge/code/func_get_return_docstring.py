from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def get_return_docstring(docstring: str):
    """Gets the docstring for a return value."""
    pattern = '\\bReturn(?:s){0,1}\\b:[ \\t\\n]*(.*?)(?=\\n|$)'
    match = re.search(pattern, docstring, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None