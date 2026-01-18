from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_class_docs_markdown(exports, docs):
    """Renders the class documentation for the package."""
    docs_classes = ''
    for class_name in exports:
        user_fn_input_type = get_deep(docs, [class_name, 'members', 'preprocess', 'return', 'type'])
        user_fn_input_description = get_deep(docs, [class_name, 'members', 'preprocess', 'return', 'description'])
        user_fn_output_type = get_deep(docs, [class_name, 'members', 'postprocess', 'value', 'type'])
        user_fn_output_description = get_deep(docs, [class_name, 'members', 'postprocess', 'value', 'description'])
        docs_classes += f'\n## `{class_name}`\n\n### Initialization\n\n{render_param_table(docs[class_name]['members']['__init__'])}\n\n{render_class_events_markdown(docs[class_name].get('events', None))}\n\n{make_user_fn_markdown(user_fn_input_type, user_fn_input_description, user_fn_output_type, user_fn_output_description)}\n'
    return docs_classes