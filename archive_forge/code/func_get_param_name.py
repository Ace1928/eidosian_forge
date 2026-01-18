from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def get_param_name(param):
    """Gets the name of a parameter."""
    if isinstance(param, str):
        return f'"{param}"'
    if inspect.isclass(param) and param.__module__ == 'builtins':
        p = getattr(param, '__name__', None)
        if p is None and inspect.isclass(param):
            p = f'{param.__module__}.{param.__name__}'
        return p
    if inspect.isclass(param):
        return f'{param.__module__}.{param.__name__}'
    param_name = getattr(param, '__name__', None)
    if param_name is None:
        param_name = str(param)
    return param_name