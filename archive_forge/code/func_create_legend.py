from __future__ import annotations
import json
from types import ModuleType
from typing import Any, Literal
from gradio_client.documentation import document
from gradio import processing_utils
from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Events
@staticmethod
def create_legend(position, title):
    if position == 'none':
        legend = None
    else:
        position = {'orient': position} if position else {}
        legend = {'title': title, **position}
    return legend