import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
def objective_expression(self) -> '_LinearExpression':
    """Returns the expression to optimize."""
    return _as_flat_linear_expression(sum((variable * self.__helper.var_objective_coefficient(variable.index) for variable in self.get_variables() if self.__helper.var_objective_coefficient(variable.index) != 0.0)) + self.__helper.objective_offset())