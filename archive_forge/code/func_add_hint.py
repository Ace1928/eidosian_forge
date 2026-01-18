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
def add_hint(self, var: IntVar, value: int) -> None:
    """Adds 'var == value' as a hint to the solver."""
    self.__model.solution_hint.vars.append(self.get_or_make_index(var))
    self.__model.solution_hint.values.append(value)