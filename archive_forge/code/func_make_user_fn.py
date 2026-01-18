from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def make_user_fn(class_name, user_fn_input_type, user_fn_input_description, user_fn_output_type, user_fn_output_description):
    """Makes the user function for the class."""
    if user_fn_input_type is None and user_fn_output_type is None and (user_fn_input_description is None) and (user_fn_output_description is None):
        return ''
    md = '\n    gr.Markdown("""\n\n### User function\n\nThe impact on the users predict function varies depending on whether the component is used as an input or output for an event (or both).\n\n- When used as an Input, the component only impacts the input signature of the user function.\n- When used as an output, the component only impacts the return signature of the user function.\n\nThe code snippet below is accurate in cases where the component is used as both an input and an output.\n\n'
    md += f'- **As input:** Is passed, {format_description(user_fn_input_description)}\n' if user_fn_input_description else ''
    md += f'- **As output:** Should return, {format_description(user_fn_output_description)}' if user_fn_output_description else ''
    if user_fn_input_type is not None or user_fn_output_type is not None:
        md += f'\n\n ```python\ndef predict(\n    value: {user_fn_input_type or 'Unknown'}\n) -> {user_fn_output_type or 'Unknown'}:\n    return value\n```'
    return f'{md}\n""", elem_classes=["md-custom", "{class_name}-user-fn"], header_links=True)\n'