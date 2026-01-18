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
def get_variable_lower_bounds(self, variables: Optional[_IndexOrSeries]=None) -> pd.Series:
    """Gets the lower bounds of all variables in the set.

        If `variables` is a `pd.Index`, then the output will be indexed by the
        variables. If `variables` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          variables (Union[pd.Index, pd.Series]): Optional. The set of variables
            from which to get the lower bounds. If unspecified, all variables will
            be in scope.

        Returns:
          pd.Series: The lower bounds of all variables in the set.
        """
    return _attribute_series(func=lambda v: v.lower_bound, values=self._get_variables(variables))