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
@contextmanager
def immediate_dispatch(doc: Document | None=None):
    """
    Context manager to trigger immediate dispatch of events triggered
    inside the execution context even when Document events are
    currently on hold.

    Arguments
    ---------
    doc: Document
        The document to dispatch events on (if `None` then `state.curdoc` is used).
    """
    doc = doc or state.curdoc
    if not doc or not doc._session_context or (not state._unblocked(doc)):
        yield
        return
    old_events = doc.callbacks._held_events
    held = doc.callbacks._hold
    doc.callbacks._held_events = []
    doc.callbacks.unhold()
    with unlocked():
        yield
    doc.callbacks._hold = held
    doc.callbacks._held_events = old_events