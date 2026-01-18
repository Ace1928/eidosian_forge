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
def process_dynamic_args(x, y, kind, **kwds):
    dynamic = {}
    arg_deps = []
    arg_names = []
    for k, v in list(kwds.items()) + [('x', x), ('y', y), ('kind', kind)]:
        if isinstance(v, param.Parameter):
            dynamic[k] = v
        elif panel_available and isinstance(v, pn.widgets.Widget):
            if Version(pn.__version__) < Version('0.6.4'):
                dynamic[k] = v.param.value
            else:
                dynamic[k] = v
    for k, v in kwds.items():
        if k not in dynamic and isinstance(v, FunctionType) and hasattr(v, '_dinfo'):
            deps = v._dinfo['dependencies']
            arg_deps += list(deps)
            arg_names += list(k) * len(deps)
    return (dynamic, arg_deps, arg_names)