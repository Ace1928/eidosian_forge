from __future__ import annotations
import asyncio
import dataclasses
import datetime as dt
import gc
import inspect
import json
import logging
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
from bokeh.application.application import SessionContext
from bokeh.core.serialization import Serializable
from bokeh.document.document import Document
from bokeh.document.events import (
from bokeh.model.util import visit_immediate_value_references
from bokeh.models import CustomJS
from ..config import config
from ..util import param_watchers
from .loading import LOADING_INDICATOR_CSS_CLASS
from .model import hold, monkeypatch_events  # noqa: F401 API import
from .state import curdoc_locked, state
def init_doc(doc: Optional[Document]) -> Document:
    curdoc = create_doc_if_none_exists(doc)
    if not curdoc.session_context:
        return curdoc
    thread_id = threading.get_ident()
    if thread_id:
        state._thread_id_[curdoc] = thread_id
    if config.global_loading_spinner:
        curdoc.js_on_event('document_ready', CustomJS(code=f"\n            const body = document.getElementsByTagName('body')[0]\n            body.classList.remove({LOADING_INDICATOR_CSS_CLASS!r}, {config.loading_spinner!r})\n            "))
    session_id = curdoc.session_context.id
    sessions = state.session_info['sessions']
    if session_id not in sessions:
        return curdoc
    sessions[session_id].update({'started': dt.datetime.now().timestamp()})
    curdoc.on_event('document_ready', state._init_session)
    return curdoc