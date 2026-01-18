from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def get_type_arguments(type_hint) -> tuple:
    """Gets the type arguments for a type hint."""
    if hasattr(type_hint, '__args__'):
        return type_hint.__args__
    elif hasattr(type_hint, '__extra__'):
        return type_hint.__extra__.__args__
    else:
        return typing.get_args(type_hint)