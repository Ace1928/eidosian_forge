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
def render_output_column(self, _submit_btn_in: Button | None) -> tuple[Button | None, ClearButton | None, DuplicateButton | None, Button | None, list | None]:
    _submit_btn = _submit_btn_in
    _clear_btn, duplicate_btn, flag_btns, _stop_btn = (None, None, None, None)
    with Column(variant='panel'):
        for component in self.output_components:
            if not isinstance(component, State):
                component.render()
        with Row():
            if self.interface_type == InterfaceTypes.OUTPUT_ONLY:
                _clear_btn = ClearButton(**self.clear_btn_params)
                _submit_btn = Button('Generate', variant='primary')
                if (inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(self.fn)) and (not self.live):
                    _stop_btn = Button(**self.stop_btn_parms)
            if self.allow_flagging == 'manual':
                flag_btns = self.render_flag_btns()
            elif self.allow_flagging == 'auto':
                if _submit_btn is None:
                    raise RenderError('Submit button not rendered')
                flag_btns = [_submit_btn]
            if self.allow_duplication:
                duplicate_btn = DuplicateButton(scale=1, size='lg', _activate=False)
    return (_submit_btn, _clear_btn, duplicate_btn, _stop_btn, flag_btns)