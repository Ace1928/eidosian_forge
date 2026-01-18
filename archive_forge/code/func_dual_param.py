import itertools
import types
import numpy as np
import pandas as pd
from .._utils import is_scalar
from ..doctools import document
from ..mapping.evaluation import after_stat
from .binning import fuzzybreaks
from .stat import stat
def dual_param(value):
    """
    Return duplicate of parameter value

    Used to apply same value to x & y axes if only one
    value is given.
    """
    if is_scalar(value):
        return types.SimpleNamespace(x=value, y=value)
    if hasattr(value, 'x') and hasattr(value, 'y'):
        return value
    if len(value) == 2:
        return types.SimpleNamespace(x=value[0], y=value[1])
    else:
        return types.SimpleNamespace(x=value, y=value)