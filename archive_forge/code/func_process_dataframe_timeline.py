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
def process_dataframe_timeline(args):
    """
    Massage input for bar traces for px.timeline()
    """
    args['is_timeline'] = True
    if args['x_start'] is None or args['x_end'] is None:
        raise ValueError('Both x_start and x_end are required')
    try:
        x_start = pd.to_datetime(args['data_frame'][args['x_start']])
        x_end = pd.to_datetime(args['data_frame'][args['x_end']])
    except (ValueError, TypeError):
        raise TypeError('Both x_start and x_end must refer to data convertible to datetimes.')
    args['data_frame'][args['x_end']] = (x_end - x_start).astype('timedelta64[ns]') / np.timedelta64(1, 'ms')
    args['x'] = args['x_end']
    del args['x_end']
    args['base'] = args['x_start']
    del args['x_start']
    return args