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
def add_map_domain(self, var: IntVar, bool_var_array: Iterable[IntVar], offset: IntegralT=0):
    """Adds `var == i + offset <=> bool_var_array[i] == true for all i`."""
    for i, bool_var in enumerate(bool_var_array):
        b_index = bool_var.index
        var_index = var.index
        model_ct = self.__model.constraints.add()
        model_ct.linear.vars.append(var_index)
        model_ct.linear.coeffs.append(1)
        model_ct.linear.domain.extend([offset + i, offset + i])
        model_ct.enforcement_literal.append(b_index)
        model_ct = self.__model.constraints.add()
        model_ct.linear.vars.append(var_index)
        model_ct.linear.coeffs.append(1)
        model_ct.enforcement_literal.append(-b_index - 1)
        if offset + i - 1 >= INT_MIN:
            model_ct.linear.domain.extend([INT_MIN, offset + i - 1])
        if offset + i + 1 <= INT_MAX:
            model_ct.linear.domain.extend([offset + i + 1, INT_MAX])