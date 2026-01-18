from __future__ import annotations
import re
import sys
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import lazy_load
from .base import ModelPane
@classmethod
def is_altair(cls, obj):
    if 'altair' in sys.modules:
        import altair as alt
        return isinstance(obj, alt.api.TopLevelMixin)
    return False