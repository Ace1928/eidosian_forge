import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
def lognorm(self, limits=None):
    """Unity-based normalization log scale.
           Apply the same transformation as matplotlib.colors.LogNorm

        Args:
            limits: tuple of (min, max) defining the normalization range
        """
    kwargs = {}
    if limits is not None:
        kwargs = {'min': limits[0], 'max': limits[1]}
    return type(self)(self, lognorm, **kwargs)