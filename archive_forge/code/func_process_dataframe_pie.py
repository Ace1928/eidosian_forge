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
def process_dataframe_pie(args, trace_patch):
    names = args.get('names')
    if names is None:
        return (args, trace_patch)
    order_in = args['category_orders'].get(names, {}).copy()
    if not order_in:
        return (args, trace_patch)
    df = args['data_frame']
    trace_patch['sort'] = False
    trace_patch['direction'] = 'clockwise'
    uniques = list(df[names].unique())
    order = [x for x in OrderedDict.fromkeys(list(order_in) + uniques) if x in uniques]
    args['data_frame'] = df.set_index(names).loc[order].reset_index()
    return (args, trace_patch)