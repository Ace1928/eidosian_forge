import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
def interface_applies(self, dataset, coerce):
    return dataset.interface.gridded and (coerce or dataset.interface.datatype == 'xarray')