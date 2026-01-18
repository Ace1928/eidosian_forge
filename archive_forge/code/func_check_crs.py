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
def check_crs(crs):
    """
    Checks if the crs represents a valid grid, projection or ESPG string.

    (Code copied and adapted from https://github.com/fmaussion/salem)

    Examples
    --------
    >>> p = check_crs('epsg:26915 +units=m')
    >>> p.srs
    '+proj=utm +zone=15 +datum=NAD83 +units=m +no_defs'
    >>> p = check_crs('wrong')
    >>> p is None
    True

    Returns
    -------
    A valid crs if possible, otherwise None.
    """
    import pyproj
    try:
        crs_type = pyproj.crs.CRS
    except AttributeError:

        class Dummy:
            pass
        crs_type = Dummy
    if isinstance(crs, pyproj.Proj):
        out = crs
    elif isinstance(crs, crs_type):
        out = pyproj.Proj(crs.to_wkt(), preserve_units=True)
    elif isinstance(crs, dict) or isinstance(crs, str):
        if isinstance(crs, str):
            try:
                crs = pyproj.CRS.from_wkt(crs)
            except RuntimeError:
                crs = crs.replace(' ', '').replace('+', ' +')
        try:
            out = pyproj.Proj(crs, preserve_units=True)
        except RuntimeError:
            try:
                out = pyproj.Proj(init=crs, preserve_units=True)
            except RuntimeError:
                out = None
    else:
        out = None
    return out