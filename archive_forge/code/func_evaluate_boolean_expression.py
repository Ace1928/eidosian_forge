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
def evaluate_boolean_expression(literal: LiteralT, solution: cp_model_pb2.CpSolverResponse) -> bool:
    """Evaluate a boolean expression against a solution."""
    if isinstance(literal, numbers.Integral):
        return bool(literal)
    elif isinstance(literal, IntVar) or isinstance(literal, _NotBooleanVariable):
        index: int = cast(Union[IntVar, _NotBooleanVariable], literal).index
        if index >= 0:
            return bool(solution.solution[index])
        else:
            return not solution.solution[-index - 1]
    else:
        raise TypeError(f'Cannot interpret {literal} as a boolean expression.')