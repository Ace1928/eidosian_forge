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
def get_bool_var_from_proto_index(self, index: int) -> IntVar:
    """Returns an already created Boolean variable from its index."""
    if index < 0 or index >= len(self.__model.variables):
        raise ValueError(f'get_bool_var_from_proto_index: out of bound index {index}')
    var = self.__model.variables[index]
    if len(var.domain) != 2 or var.domain[0] < 0 or var.domain[1] > 1:
        raise ValueError(f'get_bool_var_from_proto_index: index {index} does not reference' + ' a Boolean variable')
    return IntVar(self.__model, index, None)