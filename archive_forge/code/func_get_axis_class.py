import calendar
import datetime as dt
import re
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from itertools import permutations
import bokeh
import numpy as np
import pandas as pd
from bokeh.core.json_encoder import serialize_json  # noqa (API import)
from bokeh.core.property.datetime import Datetime
from bokeh.core.validation import silence
from bokeh.layouts import Column, Row, group_tools
from bokeh.models import (
from bokeh.models.formatters import PrintfTickFormatter, TickFormatter
from bokeh.models.scales import CategoricalScale, LinearScale, LogScale
from bokeh.models.widgets import DataTable, Div
from bokeh.plotting import figure
from bokeh.themes import built_in_themes
from bokeh.themes.theme import Theme
from packaging.version import Version
from ...core.layout import Layout
from ...core.ndmapping import NdMapping
from ...core.overlay import NdOverlay, Overlay
from ...core.spaces import DynamicMap, get_nested_dmaps
from ...core.util import (
from ...util.warnings import warn
from ..util import dim_axis_label
def get_axis_class(axis_type, range_input, dim):
    if axis_type is None:
        return (None, {})
    elif axis_type == 'linear':
        return (LinearAxis, {})
    elif axis_type == 'log':
        return (LogAxis, {})
    elif axis_type == 'datetime':
        return (DatetimeAxis, {})
    elif axis_type == 'mercator':
        return (MercatorAxis, dict(dimension='lon' if dim == 0 else 'lat'))
    elif axis_type == 'auto':
        if isinstance(range_input, FactorRange):
            return (CategoricalAxis, {})
        elif isinstance(range_input, Range1d):
            try:
                value = range_input.start
                if Datetime.is_timestamp(value):
                    return (LinearAxis, {})
                Datetime.validate(Datetime(), value)
                return (DatetimeAxis, {})
            except ValueError:
                pass
        return (LinearAxis, {})
    else:
        raise ValueError(f"Unrecognized axis_type: '{axis_type!r}'")