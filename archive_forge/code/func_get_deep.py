from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def get_deep(dictionary: dict, keys: list[str], default=None):
    """Gets a value from a nested dictionary without erroring if the key doesn't exist."""
    try:
        for key in keys:
            dictionary = dictionary[key]
        return dictionary
    except KeyError:
        return default