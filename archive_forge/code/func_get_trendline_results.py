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
def get_trendline_results(fig):
    """
    Extracts fit statistics for trendlines (when applied to figures generated with
    the `trendline` argument set to `"ols"`).

    Arguments:
        fig: the output of a `plotly.express` charting call
    Returns:
        A `pandas.DataFrame` with a column "px_fit_results" containing the `statsmodels`
        results objects, along with columns identifying the subset of the data the
        trendline was fit on.
    """
    return fig._px_trendlines