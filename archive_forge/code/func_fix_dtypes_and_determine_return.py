from inspect import signature
from math import prod
import numpy
import pandas
from pandas.api.types import is_scalar
from pandas.core.dtypes.common import is_bool_dtype, is_list_like, is_numeric_dtype
import modin.pandas as pd
from modin.core.dataframe.algebra import Binary, Map, Reduce
from modin.error_message import ErrorMessage
from .utils import try_convert_from_interoperable_type
def fix_dtypes_and_determine_return(query_compiler_in, _ndim, dtype=None, out=None, where=True):
    if dtype is not None:
        query_compiler_in = query_compiler_in.astype({col_name: dtype for col_name in query_compiler_in.columns})
    result = array(_query_compiler=query_compiler_in, _ndim=_ndim)
    if out is not None:
        out = try_convert_from_interoperable_type(out, copy=False)
        check_can_broadcast_to_output(result, out)
        result._query_compiler = result._query_compiler.astype({col_name: out.dtype for col_name in result._query_compiler.columns})
        if isinstance(where, array):
            out._update_inplace(where.where(result, out)._query_compiler)
        elif where:
            out._update_inplace(result._query_compiler)
        return out
    if isinstance(where, array) and out is None:
        from .array_creation import zeros_like
        out = zeros_like(result).astype(dtype if dtype is not None else result.dtype)
        out._query_compiler = where.where(result, out)._query_compiler
        return out
    elif not where:
        from .array_creation import zeros_like
        return zeros_like(result)
    return result