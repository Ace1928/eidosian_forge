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
def run_app(handler, module, doc, post_run=None):
    try:
        old_doc = curdoc()
    except RuntimeError:
        old_doc = None
        bk_set_curdoc(doc)
    sessions = []

    def post_check():
        newdoc = curdoc()
        if config.autoreload:
            newdoc.modules._modules = []
        if newdoc is not doc:
            raise RuntimeError("%s at '%s' replaced the output document" % (handler._origin, handler._runner.path))
    try:
        state._launching.append(doc)
        with _monkeypatch_io(handler._loggers):
            with patch_curdoc(doc):
                with profile_ctx(config.profiler) as sessions:
                    with record_modules(handler=handler):
                        handler._runner.run(module, post_check)
                        if post_run:
                            post_run()
    finally:
        if config.profiler:
            try:
                path = doc.session_context.request.path
                state._profiles[path, config.profiler] += sessions
                state.param.trigger('_profiles')
            except Exception:
                pass
        state._launching.remove(doc)
        if old_doc is not None:
            bk_set_curdoc(old_doc)