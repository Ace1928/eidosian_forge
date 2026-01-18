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
def new_bool_var(self, name: str) -> IntVar:
    """Creates a 0-1 variable with the given name."""
    return IntVar(self.__model, Domain(0, 1), name)