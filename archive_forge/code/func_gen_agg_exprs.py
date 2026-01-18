from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def gen_agg_exprs(self):
    exprs = super().gen_agg_exprs()
    for expr in exprs.values():
        expr.interpolation = self._interpolation
    return exprs