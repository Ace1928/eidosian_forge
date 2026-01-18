import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def is_streamz(data):
    if not check_library(data, 'streamz'):
        return False
    import streamz.dataframe as sdf
    return sdf and isinstance(data, (sdf.DataFrame, sdf.Series, sdf.DataFrames, sdf.Seriess))