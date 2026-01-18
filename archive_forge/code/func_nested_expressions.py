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
@_inherit_docstrings(BaseExpr.nested_expressions)
def nested_expressions(self) -> Generator[Type['BaseExpr'], Type['BaseExpr'], Type['BaseExpr']]:
    expr = (yield from super().nested_expressions())
    if (partition_keys := getattr(self, 'partition_keys', None)):
        for i, key in enumerate(partition_keys):
            new_key = (yield key)
            if new_key is not None:
                if new_key is not key:
                    if expr is self:
                        expr = self.copy()
                    expr.partition_keys[i] = new_key
                yield expr
        for i, key in enumerate(self.order_keys):
            field = key['field']
            new_field = (yield field)
            if new_field is not None:
                if new_field is not field:
                    if expr is self:
                        expr = self.copy()
                    expr.order_keys[i]['field'] = new_field
                yield expr
    return expr