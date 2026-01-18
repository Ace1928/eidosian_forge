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
def merge_tools(plot_grid, disambiguation_properties=None):
    """
    Merges tools defined on a grid of plots into a single toolbar.
    All tools of the same type are merged unless they define one
    of the disambiguation properties. By default `name`, `icon`, `tags`
    and `description` can be used to prevent tools from being merged.
    """
    tools = []
    for row in plot_grid:
        for item in row:
            if isinstance(item, LayoutDOM):
                for p in item.select(dict(type=Plot)):
                    tools.extend(p.toolbar.tools)
            if isinstance(item, GridPlot):
                item.toolbar_location = None

    def merge(tool, group):
        if issubclass(tool, (SaveTool, CopyTool, ExamineTool, FullscreenTool)):
            return tool()
        else:
            return None
    if not disambiguation_properties:
        disambiguation_properties = {'name', 'icon', 'tags', 'description'}
    ignore = set()
    for tool in tools:
        for p in tool.properties_with_values():
            if p not in disambiguation_properties:
                ignore.add(p)
    return Toolbar(tools=group_tools(tools, merge=merge, ignore=ignore) if merge_tools else tools)