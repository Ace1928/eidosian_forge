from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
def on_edit(self, callback: Callable[[TableEditEvent], None]):
    """
        Register a callback to be executed when a cell is edited.
        Whenever a cell is edited on_edit callbacks are called with
        a TableEditEvent as the first argument containing the column,
        row and value of the edited cell.

        Arguments
        ---------
        callback: (callable)
            The callback to run on edit events.
        """
    self._on_edit_callbacks.append(callback)