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
def get_linear_constraints(self) -> pd.Index:
    """Gets all linear constraints in the model."""
    return pd.Index([self.linear_constraint_from_index(i) for i in range(self.num_constraints)], name='linear_constraint')