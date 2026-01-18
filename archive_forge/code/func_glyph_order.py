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
def glyph_order(keys, draw_order=None):
    """
    Orders a set of glyph handles using regular sort and an explicit
    sort order. The explicit draw order must take the form of a list
    of glyph names while the keys should be glyph names with a custom
    suffix. The draw order may only match subset of the keys and any
    matched items will take precedence over other entries.
    """
    if draw_order is None:
        draw_order = []
    keys = sorted(keys)

    def order_fn(glyph):
        matches = [item for item in draw_order if glyph.startswith(item)]
        return (draw_order.index(matches[0]), glyph) if matches else (1000000000.0 + keys.index(glyph), glyph)
    return sorted(keys, key=order_fn)