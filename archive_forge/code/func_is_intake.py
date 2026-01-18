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
def is_intake(data):
    if 'intake' not in sys.modules:
        return False
    from intake.source.base import DataSource
    return isinstance(data, DataSource)