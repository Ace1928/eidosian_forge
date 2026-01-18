import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def set_cartesian_axis_opts(args, axis, letter, orders):
    log_key = 'log_' + letter
    range_key = 'range_' + letter
    if log_key in args and args[log_key]:
        axis['type'] = 'log'
        if range_key in args and args[range_key]:
            axis['range'] = [math.log(r, 10) for r in args[range_key]]
    elif range_key in args and args[range_key]:
        axis['range'] = args[range_key]
    if args[letter] in orders:
        axis['categoryorder'] = 'array'
        axis['categoryarray'] = orders[args[letter]] if isinstance(axis, go.layout.XAxis) else list(reversed(orders[args[letter]]))