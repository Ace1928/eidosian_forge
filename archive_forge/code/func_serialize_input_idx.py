import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def serialize_input_idx(self, expr):
    """
        Serialize ``CalciteInputIdxExpr`` expression.

        Parameters
        ----------
        expr : CalciteInputIdxExpr
            An expression to serialize.

        Returns
        -------
        int
            Serialized expression.
        """
    return expr.input