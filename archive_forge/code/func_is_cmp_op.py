import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
def is_cmp_op(op):
    """
    Check if operation is a comparison.

    Parameters
    ----------
    op : str
        Operation to check.

    Returns
    -------
    bool
        True for comparison operations and False otherwise.
    """
    return op in _cmp_ops