from __future__ import annotations
import textwrap
from contextlib import contextmanager
from typing import (
import numpy as np
from bokeh.core.serialization import Serializer
from bokeh.document import Document
from bokeh.document.events import (
from bokeh.document.json import PatchJson
from bokeh.model import DataModel
from bokeh.models import ColumnDataSource, FlexBox, Model
from bokeh.protocol.messages.patch_doc import patch_doc
from .state import state
def monkeypatch_events(events: List[DocumentChangedEvent]) -> None:
    """
    Patch events applies patches to events that are to be dispatched
    avoiding various issues in Bokeh.
    """
    for e in events:
        if isinstance(getattr(e, 'hint', None), ColumnDataChangedEvent):
            e.hint.cols = None
        elif isinstance(e, ModelChangedEvent) and isinstance(e.model, DataModel) and isinstance(e.new, np.ndarray):
            new_array = comparable_array(e.new.shape, e.new.dtype, e.new)
            e.new = new_array
            e.serializable_new = new_array