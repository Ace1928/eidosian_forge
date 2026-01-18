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
def make_axis(axis, size, factors, dim, flip=False, rotation=0, label_size=None, tick_size=None, axis_height=35):
    factors = list(map(dim.pprint_value, factors))
    nchars = np.max([len(f) for f in factors])
    ranges = FactorRange(factors=factors)
    ranges2 = Range1d(start=0, end=1)
    axis_label = dim_axis_label(dim)
    reset = 'range.setv({start: 0, end: range.factors.length})'
    customjs = CustomJS(args=dict(range=ranges), code=reset)
    ranges.js_on_change('start', customjs)
    axis_props = {}
    if label_size:
        axis_props['axis_label_text_font_size'] = label_size
    if tick_size:
        axis_props['major_label_text_font_size'] = tick_size
    tick_px = font_size_to_pixels(tick_size)
    if tick_px is None:
        tick_px = 8
    label_px = font_size_to_pixels(label_size)
    if label_px is None:
        label_px = 10
    rotation = np.radians(rotation)
    if axis == 'x':
        align = 'center'
        height = int(axis_height + np.abs(np.sin(rotation)) * (nchars * tick_px * 0.82)) + tick_px + label_px
        opts = dict(x_axis_type='auto', x_axis_label=axis_label, x_range=ranges, y_range=ranges2, height=height, width=size)
    else:
        align = 'left' if flip else 'right'
        width = int(axis_height + np.abs(np.cos(rotation)) * (nchars * tick_px * 0.82)) + tick_px + label_px
        opts = dict(y_axis_label=axis_label, x_range=ranges2, y_range=ranges, height=size, width=width)
    p = figure(toolbar_location=None, tools=[], **opts)
    p.outline_line_alpha = 0
    p.grid.grid_line_alpha = 0
    if axis == 'x':
        p.align = 'end'
        p.yaxis.visible = False
        axis = p.xaxis[0]
        if flip:
            p.above = p.below
            p.below = []
            p.xaxis[:] = p.above
    else:
        p.xaxis.visible = False
        axis = p.yaxis[0]
        if flip:
            p.right = p.left
            p.left = []
            p.yaxis[:] = p.right
    axis.major_label_orientation = rotation
    axis.major_label_text_align = align
    axis.major_label_text_baseline = 'middle'
    axis.update(**axis_props)
    return p