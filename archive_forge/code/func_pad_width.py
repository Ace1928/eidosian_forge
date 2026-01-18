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
def pad_width(model, table_padding=0.85, tabs_padding=1.2):
    """
    Computes the width of a model and sets up appropriate padding
    for Tabs and DataTable types.
    """
    if isinstance(model, Row):
        vals = [pad_width(child) for child in model.children]
        width = np.max([v for v in vals if v is not None])
    elif isinstance(model, Column):
        vals = [pad_width(child) for child in model.children]
        width = np.sum([v for v in vals if v is not None])
    elif isinstance(model, Tabs):
        vals = [pad_width(t) for t in model.tabs]
        width = np.max([v for v in vals if v is not None])
        for submodel in model.tabs:
            submodel.width = width
            width = int(tabs_padding * width)
    elif isinstance(model, DataTable):
        width = model.width
        model.width = int(table_padding * width)
    elif isinstance(model, Div):
        width = model.width
    elif model:
        width = model.width
    else:
        width = 0
    return width