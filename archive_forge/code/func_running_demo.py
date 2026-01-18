import gradio as gr
from gradio.routes import App
from gradio.utils import BaseReloader
@property
def running_demo(self):
    return self._running[self.current_cell]