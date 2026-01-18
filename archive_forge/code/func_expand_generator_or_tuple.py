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
def expand_generator_or_tuple(args):
    if hasattr(args, '__len__'):
        if len(args) != 1:
            return args
        if isinstance(args[0], (numbers.Number, LinearExpr)):
            return args
    return args[0]