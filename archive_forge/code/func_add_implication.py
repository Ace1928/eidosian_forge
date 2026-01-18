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
def add_implication(self, a: LiteralT, b: LiteralT) -> Constraint:
    """Adds `a => b` (`a` implies `b`)."""
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.bool_or.literals.append(self.get_or_make_boolean_index(b))
    model_ct.enforcement_literal.append(self.get_or_make_boolean_index(a))
    return ct