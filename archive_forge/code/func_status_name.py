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
def status_name(self, status: Optional[Any]=None) -> str:
    """Returns the name of the status returned by solve()."""
    if status is None:
        status = self._solution.status
    return cp_model_pb2.CpSolverStatus.Name(status)