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
def new_interval_var(self, start: LinearExprT, size: LinearExprT, end: LinearExprT, name: str) -> IntervalVar:
    """Creates an interval variable from start, size, and end.

        An interval variable is a constraint, that is itself used in other
        constraints like NoOverlap.

        Internally, it ensures that `start + size == end`.

        Args:
          start: The start of the interval. It must be of the form a * var + b.
          size: The size of the interval. It must be of the form a * var + b.
          end: The end of the interval. It must be of the form a * var + b.
          name: The name of the interval variable.

        Returns:
          An `IntervalVar` object.
        """
    lin = self.add(start + size == end)
    if name:
        lin.with_name('lin_' + name)
    start_expr = self.parse_linear_expression(start)
    size_expr = self.parse_linear_expression(size)
    end_expr = self.parse_linear_expression(end)
    if len(start_expr.vars) > 1:
        raise TypeError('cp_model.new_interval_var: start must be 1-var affine or constant.')
    if len(size_expr.vars) > 1:
        raise TypeError('cp_model.new_interval_var: size must be 1-var affine or constant.')
    if len(end_expr.vars) > 1:
        raise TypeError('cp_model.new_interval_var: end must be 1-var affine or constant.')
    return IntervalVar(self.__model, start_expr, size_expr, end_expr, None, name)