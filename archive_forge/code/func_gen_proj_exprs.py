from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def gen_proj_exprs(self):
    """
            Generate values required for intermediate aggregates computation.

            Returns
            -------
            dict
                New column expressions mapped to their names.
            """
    quad_expr = self._builder._translate(self._arg.mul(self._arg))
    cube_expr = self._builder._translate(self._arg.mul(self._arg).mul(self._arg))
    return {self._quad_name: quad_expr, self._cube_name: cube_expr}