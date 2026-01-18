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
def get_or_make_index_from_constant(self, value: IntegralT) -> int:
    if value in self.__constant_map:
        return self.__constant_map[value]
    index = len(self.__model.variables)
    self.__model.variables.add(domain=[value, value])
    self.__constant_map[value] = index
    return index