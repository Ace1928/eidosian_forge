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
def new_optional_interval_var(self, start: LinearExprT, size: LinearExprT, end: LinearExprT, is_present: LiteralT, name: str) -> IntervalVar:
    """Creates an optional interval var from start, size, end, and is_present.

        An optional interval variable is a constraint, that is itself used in other
        constraints like NoOverlap. This constraint is protected by a presence
        literal that indicates if it is active or not.

        Internally, it ensures that `is_present` implies `start + size ==
        end`.

        Args:
          start: The start of the interval. It must be of the form a * var + b.
          size: The size of the interval. It must be of the form a * var + b.
          end: The end of the interval. It must be of the form a * var + b.
          is_present: A literal that indicates if the interval is active or not. A
            inactive interval is simply ignored by all constraints.
          name: The name of the interval variable.

        Returns:
          An `IntervalVar` object.
        """
    lin = self.add(start + size == end).only_enforce_if(is_present)
    if name:
        lin.with_name('lin_opt_' + name)
    is_present_index = self.get_or_make_boolean_index(is_present)
    start_expr = self.parse_linear_expression(start)
    size_expr = self.parse_linear_expression(size)
    end_expr = self.parse_linear_expression(end)
    if len(start_expr.vars) > 1:
        raise TypeError('cp_model.new_interval_var: start must be affine or constant.')
    if len(size_expr.vars) > 1:
        raise TypeError('cp_model.new_interval_var: size must be affine or constant.')
    if len(end_expr.vars) > 1:
        raise TypeError('cp_model.new_interval_var: end must be affine or constant.')
    return IntervalVar(self.__model, start_expr, size_expr, end_expr, is_present_index, name)