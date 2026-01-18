import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def get_or_make_index(self, arg: VariableT) -> int:
    """Returns the index of a variable, its negation, or a number."""
    if isinstance(arg, IntVar):
        return arg.index
    if isinstance(arg, _ProductCst) and isinstance(arg.expression(), IntVar) and (arg.coefficient() == -1):
        return -arg.expression().index - 1
    if isinstance(arg, numbers.Integral):
        arg = cmh.assert_is_int64(arg)
        return self.get_or_make_index_from_constant(arg)
    raise TypeError('NotSupported: model.get_or_make_index(' + str(arg) + ')')