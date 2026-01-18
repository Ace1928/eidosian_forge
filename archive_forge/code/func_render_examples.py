from __future__ import annotations
import inspect
import json
import os
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Literal
from gradio_client.documentation import document
from gradio import Examples, utils
from gradio.blocks import Blocks
from gradio.components import (
from gradio.data_classes import InterfaceTypes
from gradio.events import Dependency, Events, on
from gradio.exceptions import RenderError
from gradio.flagging import CSVLogger, FlaggingCallback, FlagMethod
from gradio.layouts import Accordion, Column, Row, Tab, Tabs
from gradio.pipelines import load_from_pipeline
from gradio.themes import ThemeClass as Theme
def render_examples(self):
    if self.examples:
        non_state_inputs = [c for c in self.input_components if not isinstance(c, State)]
        non_state_outputs = [c for c in self.output_components if not isinstance(c, State)]
        self.examples_handler = Examples(examples=self.examples, inputs=non_state_inputs, outputs=non_state_outputs, fn=self.fn, cache_examples=self.cache_examples, examples_per_page=self.examples_per_page, _api_mode=self.api_mode, batch=self.batch)