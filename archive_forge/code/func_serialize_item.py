import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def serialize_item(self, item):
    """
        Serialize a single expression item.

        Parameters
        ----------
        item : Any
            Item to serialize.

        Returns
        -------
        str, int, None, dict or list of dict
            Serialized item.
        """
    if isinstance(item, CalciteBaseNode):
        return self.serialize_node(item)
    elif isinstance(item, BaseExpr):
        return self.serialize_expr(item)
    elif isinstance(item, CalciteCollation):
        return self.serialize_obj(item)
    elif isinstance(item, list):
        return [self.serialize_item(v) for v in item]
    elif isinstance(item, dict):
        return {k: self.serialize_item(v) for k, v in item.items()}
    self.expect_one_of(item, str, int, type(None))
    return item