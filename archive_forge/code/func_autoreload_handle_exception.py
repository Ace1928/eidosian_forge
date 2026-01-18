from __future__ import annotations
import ast
import html
import json
import logging
import os
import pathlib
import re
import sys
import traceback
import urllib.parse as urlparse
from contextlib import contextmanager
from types import ModuleType
from typing import IO, Any, Callable
import bokeh.command.util
from bokeh.application.handlers.code import CodeHandler
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler, handle_exception
from bokeh.core.types import PathLike
from bokeh.document import Document
from bokeh.io.doc import curdoc, patch_curdoc, set_curdoc as bk_set_curdoc
from bokeh.util.dependencies import import_required
from ..config import config
from .mime_render import MIME_RENDERERS
from .profile import profile_ctx
from .reload import record_modules
from .state import state
def autoreload_handle_exception(handler, module, e):
    if not config.autoreload:
        handle_exception(handler, e)
        return
    from ..pane import Alert
    del sys.modules[module.__name__]
    state.curdoc.modules._modules.remove(module)
    e_msg = str(e).replace('\x1b[1m', '<b>').replace('\x1b[0m', '</b>')
    tb = html.escape(traceback.format_exc()).replace('\x1b[1m', '<b>').replace('\x1b[0m', '</b>')
    Alert(f'<b>{type(e).__name__}</b>: {e_msg}\n<pre style="overflow-y: auto">{tb}</pre>', alert_type='danger', margin=5, sizing_mode='stretch_width').servable()