import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def load_color(color_name):
    color = [color for color in colors if color.name == color_name][0]
    return [getattr(color, f'c{i}') for i in palette_range]