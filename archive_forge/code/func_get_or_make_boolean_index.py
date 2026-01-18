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
def get_or_make_boolean_index(self, arg: LiteralT) -> int:
    """Returns an index from a boolean expression."""
    if isinstance(arg, IntVar):
        self.assert_is_boolean_variable(arg)
        return arg.index
    if isinstance(arg, _NotBooleanVariable):
        self.assert_is_boolean_variable(arg.negated())
        return arg.index
    if isinstance(arg, numbers.Integral):
        arg = cmh.assert_is_zero_or_one(arg)
        return self.get_or_make_index_from_constant(arg)
    if cmh.is_boolean(arg):
        return self.get_or_make_index_from_constant(int(arg))
    raise TypeError(f'not supported: model.get_or_make_boolean_index({arg})')