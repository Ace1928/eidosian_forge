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
def add_linear_expression_in_domain(self, linear_expr: LinearExprT, domain: Domain) -> Constraint:
    """Adds the constraint: `linear_expr` in `domain`."""
    if isinstance(linear_expr, LinearExpr):
        ct = Constraint(self)
        model_ct = self.__model.constraints[ct.index]
        coeffs_map, constant = linear_expr.get_integer_var_value_map()
        for t in coeffs_map.items():
            if not isinstance(t[0], IntVar):
                raise TypeError('Wrong argument' + str(t))
            c = cmh.assert_is_int64(t[1])
            model_ct.linear.vars.append(t[0].index)
            model_ct.linear.coeffs.append(c)
        model_ct.linear.domain.extend([cmh.capped_subtraction(x, constant) for x in domain.flattened_intervals()])
        return ct
    if isinstance(linear_expr, numbers.Integral):
        if not domain.contains(int(linear_expr)):
            return self.add_bool_or([])
        else:
            return self.add_bool_and([])
    raise TypeError('not supported: CpModel.add_linear_expression_in_domain(' + str(linear_expr) + ' ' + str(domain) + ')')