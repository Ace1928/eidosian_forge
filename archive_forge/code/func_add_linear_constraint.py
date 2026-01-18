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
def add_linear_constraint(self, linear_expr: LinearExprT, lb: IntegralT, ub: IntegralT) -> Constraint:
    """Adds the constraint: `lb <= linear_expr <= ub`."""
    return self.add_linear_expression_in_domain(linear_expr, Domain(lb, ub))