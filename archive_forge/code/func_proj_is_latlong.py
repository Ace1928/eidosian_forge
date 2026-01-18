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
def proj_is_latlong(proj):
    """Shortcut function because of deprecation."""
    try:
        return proj.is_latlong()
    except AttributeError:
        return proj.crs.is_geographic