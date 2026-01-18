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
def new_int_var(self, lb: IntegralT, ub: IntegralT, name: str) -> IntVar:
    """Create an integer variable with domain [lb, ub].

        The CP-SAT solver is limited to integer variables. If you have fractional
        values, scale them up so that they become integers; if you have strings,
        encode them as integers.

        Args:
          lb: Lower bound for the variable.
          ub: Upper bound for the variable.
          name: The name of the variable.

        Returns:
          a variable whose domain is [lb, ub].
        """
    return IntVar(self.__model, Domain(lb, ub), name)