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
def is_polars(data):
    if not check_library(data, 'polars'):
        return False
    import polars as pl
    return isinstance(data, (pl.DataFrame, pl.Series, pl.LazyFrame))