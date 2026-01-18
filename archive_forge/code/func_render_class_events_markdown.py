from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def render_class_events_markdown(events):
    """Renders the events for a class."""
    if len(events) == 0:
        return ''
    event_table = '\n### Events\n\n| name | description |\n|:-----|:------------|\n'
    for event_name, event in events.items():
        event_table += f'| `{event_name}` | {event['description']} |\n'
    return event_table