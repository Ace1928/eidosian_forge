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
def new_num_var_series(self, name: str, index: pd.Index, lower_bounds: Union[NumberT, pd.Series]=-math.inf, upper_bounds: Union[NumberT, pd.Series]=math.inf) -> pd.Series:
    """Creates a series of continuous variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          lower_bounds (Union[int, float, pd.Series]): Optional. A lower bound for
            variables in the set. If a `pd.Series` is passed in, it will be based on
            the corresponding values of the pd.Series. Defaults to -inf.
          upper_bounds (Union[int, float, pd.Series]): Optional. An upper bound for
            variables in the set. If a `pd.Series` is passed in, it will be based on
            the corresponding values of the pd.Series. Defaults to +inf.

        Returns:
          pd.Series: The variable set indexed by its corresponding dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the `lowerbound` is greater than the `upperbound`.
          ValueError: if the index of `lower_bound`, `upper_bound`, or `is_integer`
          does not match the input index.
        """
    return self.new_var_series(name, index, lower_bounds, upper_bounds, False)