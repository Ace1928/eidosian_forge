import gradio as gr
from gradio.routes import App
from gradio.utils import BaseReloader
@property
def current_cell(self):
    return self._cell_tracker.current_cell