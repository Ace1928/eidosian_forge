import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def load_size(size_name):
    size = [size for size in sizes if size.name == size_name][0]
    return [getattr(size, i) for i in size_range]