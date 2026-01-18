from __future__ import annotations
import re
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import join_keys, match
from ..exceptions import PlotnineError, PlotnineWarning
from .facet import (
from .strips import Strips, strip
def n_to_nrow_ncol(n: int) -> tuple[int, int]:
    """
    Compute the rows and columns given the number of plots.
    """
    if n <= 3:
        nrow, ncol = (1, n)
    elif n <= 6:
        nrow, ncol = (2, (n + 1) // 2)
    elif n <= 12:
        nrow, ncol = (3, (n + 2) // 3)
    else:
        ncol = int(np.ceil(np.sqrt(n)))
        nrow = int(np.ceil(n / ncol))
    return (nrow, ncol)