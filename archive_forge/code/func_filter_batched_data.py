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
def filter_batched_data(data, mapping):
    """
    Iterates over the data and mapping for a ColumnDataSource and
    replaces columns with repeating values with a scalar. This is
    purely and optimization for scalar types.
    """
    for k, v in list(mapping.items()):
        if isinstance(v, dict) and 'field' in v:
            if 'transform' in v:
                continue
            v = v['field']
        elif not isinstance(v, str):
            continue
        values = data[v]
        try:
            if len(unique_array(values)) == 1:
                mapping[k] = values[0]
                del data[v]
        except Exception:
            pass