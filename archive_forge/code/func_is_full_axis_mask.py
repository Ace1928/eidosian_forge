import logging
import uuid
from abc import ABC
from copy import copy
import pandas
from pandas.api.types import is_scalar
from pandas.util import cache_readonly
from modin.core.storage_formats.pandas.utils import length_fn_pandas, width_fn_pandas
from modin.logging import ClassLogger, get_logger
from modin.pandas.indexing import compute_sliced_len
def is_full_axis_mask(index, axis_length):
    """Check whether `index` mask grabs `axis_length` amount of elements."""
    if isinstance(index, slice):
        return index == slice(None) or (isinstance(axis_length, int) and compute_sliced_len(index, axis_length) == axis_length)
    return hasattr(index, '__len__') and isinstance(axis_length, int) and (len(index) == axis_length)