import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def expect_one_of(self, val, *types):
    """
        Raise an error if values doesn't belong to any of specified types.

        Parameters
        ----------
        val : Any
            Value to check.
        *types : list of type
            Allowed value types.
        """
    for t in types:
        if isinstance(val, t):
            return
    raise TypeError('Can not serialize {}'.format(type(val).__name__))