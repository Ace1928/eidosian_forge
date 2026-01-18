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
def filter_opts(eltype, options, backend='bokeh'):
    opts = getattr(hv.Store.options(backend), eltype)
    allowed = [k for g in opts.groups.values() for k in list(g.allowed_keywords)]
    opts = {k: v for k, v in options.items() if k in allowed}
    return opts