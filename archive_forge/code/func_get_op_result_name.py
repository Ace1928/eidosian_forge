from __future__ import annotations
from functools import wraps
from typing import (
from pandas._libs.lib import item_from_zerodim
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.generic import (
def get_op_result_name(left, right):
    """
    Find the appropriate name to pin to an operation result.  This result
    should always be either an Index or a Series.

    Parameters
    ----------
    left : {Series, Index}
    right : object

    Returns
    -------
    name : object
        Usually a string
    """
    if isinstance(right, (ABCSeries, ABCIndex)):
        name = _maybe_match_name(left, right)
    else:
        name = left.name
    return name