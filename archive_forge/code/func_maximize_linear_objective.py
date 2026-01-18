import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def maximize_linear_objective(self, objective: LinearTypes) -> None:
    """Sets the objective to maximize the provided linear expression `objective`."""
    self.set_linear_objective(objective, is_maximize=True)