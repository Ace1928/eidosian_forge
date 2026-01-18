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
def sync_legends(bokeh_layout):
    """This syncs the legends of all plots in a grid based on their name.

    Parameters
    ----------
    bokeh_layout : bokeh.models.{GridPlot, Row, Column}
        Gridplot to sync legends of.
    """
    if len(bokeh_layout.children) < 2:
        return
    items = defaultdict(list)
    click_policies = set()
    for fig in bokeh_layout.children:
        if isinstance(fig, tuple):
            fig = fig[0]
        if not isinstance(fig, figure):
            continue
        for r in fig.renderers:
            if r.name:
                items[r.name].append(r)
        if fig.legend:
            click_policies.add(fig.legend[0].click_policy)
    click_policies.discard('none')
    if len(click_policies) > 1:
        warn('Click policy of legends are not the same, no syncing will happen.')
        return
    elif not click_policies:
        return
    mapping = {'mute': 'muted', 'hide': 'visible'}
    policy = mapping.get(next(iter(click_policies)))
    code = f'dst.{policy} = src.{policy}'
    for item in items.values():
        for src, dst in permutations(item, 2):
            src.js_on_change(policy, CustomJS(code=code, args=dict(src=src, dst=dst)))