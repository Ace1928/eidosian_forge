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
def as_linear_expression(self) -> LinearExpression:
    if any(self.quadratic_terms()):
        raise TypeError('Cannot get a quadratic objective as a linear expression')
    return as_flat_linear_expression(self.offset + LinearSum(self.linear_terms()))