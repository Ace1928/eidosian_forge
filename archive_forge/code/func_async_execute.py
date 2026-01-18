from __future__ import annotations
import asyncio
import datetime as dt
import importlib
import inspect
import logging
import os
import pathlib
import signal
import sys
import threading
import uuid
from contextlib import contextmanager
from functools import partial, wraps
from html import escape
from types import FunctionType, MethodType
from typing import (
from urllib.parse import urljoin, urlparse
import bokeh
import param
import tornado
from bokeh.application import Application as BkApplication
from bokeh.application.handlers.function import FunctionHandler
from bokeh.core.json_encoder import serialize_json
from bokeh.core.templates import AUTOLOAD_JS, FILE, MACROS
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
from bokeh.embed.bundle import Script
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import RenderItem
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.models import CustomJS
from bokeh.server.server import Server as BokehServer
from bokeh.server.urls import per_app_patterns, toplevel_patterns
from bokeh.server.views.autoload_js_handler import (
from bokeh.server.views.doc_handler import DocHandler as BkDocHandler
from bokeh.server.views.root_handler import RootHandler as BkRootHandler
from bokeh.server.views.static_handler import StaticHandler
from bokeh.util.serialization import make_id
from bokeh.util.token import (
from tornado.ioloop import IOLoop
from tornado.web import (
from tornado.wsgi import WSGIContainer
from ..config import config
from ..util import edit_readonly, fullpath
from ..util.warnings import warn
from .application import Application, build_single_handler_application
from .document import (  # noqa
from .liveness import LivenessHandler
from .loading import LOADING_INDICATOR_CSS_CLASS
from .logging import (
from .reload import record_modules
from .resources import (
from .session import generate_session
from .state import set_curdoc, state
def async_execute(func: Callable[..., None]) -> None:
    """
    Wrap async event loop scheduling to ensure that with_lock flag
    is propagated from function to partial wrapping it.
    """
    if not state.curdoc or not state.curdoc.session_context:
        ioloop = IOLoop.current()
        event_loop = ioloop.asyncio_loop
        wrapper = state._handle_exception_wrapper(func)
        if event_loop.is_running():
            ioloop.add_callback(wrapper)
        else:
            event_loop.run_until_complete(wrapper())
        return
    if isinstance(func, partial) and hasattr(func.func, 'lock'):
        unlock = not func.func.lock
    else:
        unlock = not getattr(func, 'lock', False)
    curdoc = state.curdoc

    @wraps(func)
    async def wrapper(*args, **kw):
        with set_curdoc(curdoc):
            try:
                return await func(*args, **kw)
            except Exception as e:
                state._handle_exception(e)
    if unlock:
        wrapper.nolock = True
    state.curdoc.add_next_tick_callback(wrapper)