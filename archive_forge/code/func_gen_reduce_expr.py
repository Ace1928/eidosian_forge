from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def gen_reduce_expr(self):
    return OpExpr('PG_UNNEST', [super().gen_reduce_expr()], self._dtype)