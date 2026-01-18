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
def render_input_column(self) -> tuple[Button | None, ClearButton | None, Button | None, list[Button] | None, Column]:
    _submit_btn, _clear_btn, _stop_btn, flag_btns = (None, None, None, None)
    with Column(variant='panel'):
        input_component_column = Column()
        with input_component_column:
            for component in self.main_input_components:
                component.render()
            if self.additional_input_components:
                with Accordion(**self.additional_inputs_accordion_params):
                    for component in self.additional_input_components:
                        component.render()
        with Row():
            if self.interface_type in [InterfaceTypes.STANDARD, InterfaceTypes.INPUT_ONLY]:
                _clear_btn = ClearButton(**self.clear_btn_params)
                if not self.live:
                    _submit_btn = Button(**self.submit_btn_parms)
                    if inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(self.fn):
                        _stop_btn = Button(**self.stop_btn_parms)
            elif self.interface_type == InterfaceTypes.UNIFIED:
                _clear_btn = ClearButton(**self.clear_btn_params)
                _submit_btn = Button(**self.submit_btn_parms)
                if (inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(self.fn)) and (not self.live):
                    _stop_btn = Button(**self.stop_btn_parms)
                if self.allow_flagging == 'manual':
                    flag_btns = self.render_flag_btns()
                elif self.allow_flagging == 'auto':
                    flag_btns = [_submit_btn]
    return (_submit_btn, _clear_btn, _stop_btn, flag_btns, input_component_column)