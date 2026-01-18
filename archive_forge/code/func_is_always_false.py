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
def is_always_false(self) -> bool:
    """Returns True if the constraint is always false.

        Usually, it means that it was created by model.add(False)
        """
    return self.lower_bound > self.upper_bound