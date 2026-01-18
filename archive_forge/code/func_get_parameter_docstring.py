from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def get_parameter_docstring(docstring: str, parameter_name: str):
    """Gets the docstring for a parameter."""
    pattern = f'\\b{parameter_name}\\b:[ \\t]*(.*?)(?=\\n|$)'
    match = re.search(pattern, docstring, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None