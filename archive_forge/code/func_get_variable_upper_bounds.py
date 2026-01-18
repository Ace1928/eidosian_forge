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
def get_variable_upper_bounds(self, variables: Optional[_IndexOrSeries]=None) -> pd.Series:
    """Gets the upper bounds of all variables in the set.

        Args:
          variables (Union[pd.Index, pd.Series]): Optional. The set of variables
            from which to get the upper bounds. If unspecified, all variables will
            be in scope.

        Returns:
          pd.Series: The upper bounds of all variables in the set.
        """
    return _attribute_series(func=lambda v: v.upper_bound, values=self._get_variables(variables))