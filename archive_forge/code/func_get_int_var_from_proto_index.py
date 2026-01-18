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
def get_int_var_from_proto_index(self, index: int) -> IntVar:
    """Returns an already created integer variable from its index."""
    if index < 0 or index >= len(self.__model.variables):
        raise ValueError(f'get_int_var_from_proto_index: out of bound index {index}')
    return IntVar(self.__model, index, None)