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
def instantiate_crs_str(crs_str: str, **kwargs):
    """
    Instantiate a cartopy.crs.Projection from a string.
    """
    import cartopy.crs as ccrs
    if crs_str.upper() == 'GOOGLE_MERCATOR':
        return ccrs.GOOGLE_MERCATOR
    return getattr(ccrs, crs_str)(**kwargs)