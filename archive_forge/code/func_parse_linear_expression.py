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
def parse_linear_expression(self, linear_expr: LinearExprT, negate: bool=False) -> cp_model_pb2.LinearExpressionProto:
    """Returns a LinearExpressionProto built from a LinearExpr instance."""
    result: cp_model_pb2.LinearExpressionProto = cp_model_pb2.LinearExpressionProto()
    mult = -1 if negate else 1
    if isinstance(linear_expr, numbers.Integral):
        result.offset = int(linear_expr) * mult
        return result
    if isinstance(linear_expr, IntVar):
        result.vars.append(self.get_or_make_index(linear_expr))
        result.coeffs.append(mult)
        return result
    coeffs_map, constant = cast(LinearExpr, linear_expr).get_integer_var_value_map()
    result.offset = constant * mult
    for t in coeffs_map.items():
        if not isinstance(t[0], IntVar):
            raise TypeError('Wrong argument' + str(t))
        c = cmh.assert_is_int64(t[1])
        result.vars.append(t[0].index)
        result.coeffs.append(c * mult)
    return result