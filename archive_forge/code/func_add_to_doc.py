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
def add_to_doc(obj: Model, doc: Document, hold: bool=False, skip: Set[Model] | None=None):
    """
    Adds a model to the supplied Document removing it from any existing Documents.
    """
    models = remove_root(obj, skip=skip)
    doc.add_root(obj)
    if doc.callbacks.hold_value is None and hold:
        doc.hold()
    return models