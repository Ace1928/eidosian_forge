from __future__ import annotations
import decimal
import operator
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
from pandas.core.indexers import validate_indices
def nunique_ints(values: ArrayLike) -> int:
    """
    Return the number of unique values for integer array-likes.

    Significantly faster than pandas.unique for long enough sequences.
    No checks are done to ensure input is integral.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    int : The number of unique values in ``values``
    """
    if len(values) == 0:
        return 0
    values = _ensure_data(values)
    result = (np.bincount(values.ravel().astype('intp')) != 0).sum()
    return result