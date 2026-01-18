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
def get_tab_title(key, frame, overlay):
    """
    Computes a title for bokeh tabs from the key in the overlay, the
    element and the containing (Nd)Overlay.
    """
    if isinstance(overlay, Overlay):
        if frame is not None:
            title = []
            if frame.label:
                title.append(frame.label)
                if frame.group != frame.param.objects('existing')['group'].default:
                    title.append(frame.group)
            else:
                title.append(frame.group)
        else:
            title = key
        title = ' '.join(title)
    else:
        title = ' | '.join([d.pprint_value_string(k) for d, k in zip(overlay.kdims, key)])
    return title