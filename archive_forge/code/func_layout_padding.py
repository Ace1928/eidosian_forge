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
def layout_padding(plots, renderer):
    """
    Pads Nones in a list of lists of plots with empty plots.
    """
    widths, heights = (defaultdict(int), defaultdict(int))
    for r, row in enumerate(plots):
        for c, p in enumerate(row):
            if p is not None:
                width, height = renderer.get_size(p)
                widths[c] = max(widths[c], width)
                heights[r] = max(heights[r], height)
    expanded_plots = []
    for r, row in enumerate(plots):
        expanded_plots.append([])
        for c, p in enumerate(row):
            if p is None:
                p = empty_plot(widths[c], heights[r])
            elif hasattr(p, 'width') and p.width == 0 and (p.height == 0):
                p.width = widths[c]
                p.height = heights[r]
            expanded_plots[r].append(p)
    return expanded_plots