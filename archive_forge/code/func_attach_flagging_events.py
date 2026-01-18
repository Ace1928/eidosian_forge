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
def attach_flagging_events(self, flag_btns: list[Button] | None, _clear_btn: ClearButton, _submit_event: Dependency):
    if not (flag_btns and self.interface_type in (InterfaceTypes.STANDARD, InterfaceTypes.OUTPUT_ONLY, InterfaceTypes.UNIFIED)):
        return
    if self.allow_flagging == 'auto':
        flag_method = FlagMethod(self.flagging_callback, '', '', visual_feedback=False)
        _submit_event.success(flag_method, inputs=self.input_components + self.output_components, outputs=None, preprocess=False, queue=False, show_api=False)
        return
    if self.interface_type == InterfaceTypes.UNIFIED:
        flag_components = self.input_components
    else:
        flag_components = self.input_components + self.output_components
    for flag_btn, (label, value) in zip(flag_btns, self.flagging_options):
        if not isinstance(value, str):
            raise TypeError(f'Flagging option value must be a string, not {value!r}')
        flag_method = FlagMethod(self.flagging_callback, label, value)
        flag_btn.click(utils.async_lambda(lambda: Button(value='Saving...', interactive=False)), None, flag_btn, queue=False, show_api=False)
        flag_btn.click(flag_method, inputs=flag_components, outputs=flag_btn, preprocess=False, queue=False, show_api=False)
        _clear_btn.click(utils.async_lambda(flag_method.reset), None, flag_btn, queue=False, show_api=False)