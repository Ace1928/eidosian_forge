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
def date_to_integer(date):
    """Converts support date types to milliseconds since epoch

    Attempts highest precision conversion of different datetime
    formats to milliseconds since the epoch (1970-01-01 00:00:00).
    If datetime is a cftime with a non-standard calendar the
    caveats described in hv.core.util.cftime_to_timestamp apply.

    Args:
        date: Date- or datetime-like object

    Returns:
        Milliseconds since 1970-01-01 00:00:00
    """
    if isinstance(date, pd.Timestamp):
        try:
            date = date.to_datetime64()
        except Exception:
            date = date.to_datetime()
    if isinstance(date, np.datetime64):
        return date.astype('datetime64[ms]').astype(float)
    elif isinstance(date, cftime_types):
        return cftime_to_timestamp(date, 'ms')
    if hasattr(date, 'timetuple'):
        dt_int = calendar.timegm(date.timetuple()) * 1000
    else:
        raise ValueError('Datetime type not recognized')
    return dt_int