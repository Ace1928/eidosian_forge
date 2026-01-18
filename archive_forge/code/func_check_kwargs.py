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
def check_kwargs(order='C', subok=True, keepdims=None, casting='same_kind', where=True):
    if order not in ['K', 'C']:
        ErrorMessage.single_warning("Array order besides 'C' is not currently supported in Modin. Defaulting to 'C' order.")
    if not subok:
        ErrorMessage.single_warning('Subclassing types is not currently supported in Modin. Defaulting to the same base dtype.')
    if keepdims:
        ErrorMessage.single_warning('Modin does not yet support broadcasting between nested 1D arrays and 2D arrays.')
    if casting != 'same_kind':
        ErrorMessage.single_warning('Modin does not yet support the `casting` argument.')
    if not (is_scalar(where) or (isinstance(where, array) and is_bool_dtype(where.dtype))):
        if not isinstance(where, array):
            raise NotImplementedError(f'Modin only supports scalar or modin.numpy.array `where` parameter, not `where` parameter of type {type(where)}')
        raise TypeError(f"Cannot cast array data from {where.dtype} to dtype('bool') according to the rule 'safe'")