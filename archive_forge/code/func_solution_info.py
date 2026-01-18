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
def solution_info(self) -> str:
    """Returns some information on the solve process.

        Returns some information on how the solution was found, or the reason
        why the model or the parameters are invalid.

        Raises:
          RuntimeError: if solve() has not been called.
        """
    return self._solution.solution_info