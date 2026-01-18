from typing import Optional
import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
def is_int_index(index: pd.Index) -> bool:
    """
    Check if an index is integral

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if is an index with a standard integral type
    """
    return isinstance(index, pd.Index) and isinstance(index.dtype, np.dtype) and np.issubdtype(index.dtype, np.integer)