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
def process_derived_datetime_xarray(data, not_found):
    from pandas.api.types import is_datetime64_any_dtype as isdate
    extra_vars = []
    extra_coords = []
    for var in not_found:
        if '.' in var:
            derived_from = var.split('.')[0]
            if isdate(data[derived_from]):
                if derived_from in data.coords:
                    extra_coords.append(var)
                else:
                    extra_vars.append(var)
    not_found = [var for var in not_found if var not in extra_vars + extra_coords]
    return (not_found, extra_vars, extra_coords)