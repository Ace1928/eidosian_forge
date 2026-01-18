from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def find_first_non_return_key(some_dict):
    """Finds the first key in a dictionary that is not "return"."""
    for key, value in some_dict.items():
        if key != 'return':
            return value
    return None