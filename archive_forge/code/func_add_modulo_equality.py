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
def add_modulo_equality(self, target: LinearExprT, expr: LinearExprT, mod: LinearExprT) -> Constraint:
    """Adds `target = expr % mod`."""
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.int_mod.exprs.append(self.parse_linear_expression(expr))
    model_ct.int_mod.exprs.append(self.parse_linear_expression(mod))
    model_ct.int_mod.target.CopyFrom(self.parse_linear_expression(target))
    return ct