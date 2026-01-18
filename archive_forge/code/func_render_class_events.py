from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_class_events(events: dict, name):
    """Renders the events for a class."""
    if len(events) == 0:
        return ''
    else:
        return f'\n    gr.Markdown("### Events")\n    gr.ParamViewer(value=_docs["{name}"]["events"], linkify={['Event']})\n\n'