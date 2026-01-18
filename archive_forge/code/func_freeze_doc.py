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
def freeze_doc(doc: Document, model: HasProps, properties: Dict[str, Any], force: bool=False):
    """
    Freezes the document model references if any of the properties
    are themselves a model.
    """
    if force:
        dirty_count = 1
    else:
        dirty_count = 0

        def mark_dirty(_: HasProps):
            nonlocal dirty_count
            dirty_count += 1
        for key, value in properties.items():
            visit_immediate_value_references(getattr(model, key, None), mark_dirty)
            visit_immediate_value_references(value, mark_dirty)
    if dirty_count:
        doc.models._push_freeze()
    yield
    if dirty_count:
        doc.models._pop_freeze()