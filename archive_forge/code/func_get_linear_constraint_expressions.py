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
def get_linear_constraint_expressions(self, constraints: Optional[_IndexOrSeries]=None) -> pd.Series:
    """Gets the expressions of all linear constraints in the set.

        If `constraints` is a `pd.Index`, then the output will be indexed by the
        constraints. If `constraints` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          constraints (Union[pd.Index, pd.Series]): Optional. The set of linear
            constraints from which to get the expressions. If unspecified, all
            linear constraints will be in scope.

        Returns:
          pd.Series: The expressions of all linear constraints in the set.
        """
    return _attribute_series(func=lambda c: _as_flat_linear_expression(sum((coeff * Variable(self.__helper, var_id, None, None, None) for var_id, coeff in zip(c.helper.constraint_var_indices(c.index), c.helper.constraint_coefficients(c.index))))), values=self._get_linear_constraints(constraints))