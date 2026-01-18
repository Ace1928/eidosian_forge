import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def get_docstr(var):
    for parameters in docs_theme_core + docs_theme_vars:
        if parameters['name'] == var:
            return parameters['doc']
    raise ValueError(f'Variable {var} not found in theme documentation.')