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
def is_logical_op(op):
    """
    Check if operation is a logical one.

    Parameters
    ----------
    op : str
        Operation to check.

    Returns
    -------
    bool
        True for logical operations and False otherwise.
    """
    return op in _logical_ops