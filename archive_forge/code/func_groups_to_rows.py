import copy
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
from .compat import SKLEARN_INSTALLED, DataFrame, XGBStratifiedKFold
from .core import (
def groups_to_rows(groups: List[np.ndarray], boundaries: np.ndarray) -> np.ndarray:
    """
    Given group row boundaries, convert ground indexes to row indexes
    :param groups: list of groups for testing
    :param boundaries: rows index limits of each group
    :return: row in group
    """
    return np.concatenate([np.arange(boundaries[g], boundaries[g + 1]) for g in groups])