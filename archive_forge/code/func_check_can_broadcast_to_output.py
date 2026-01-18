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
def check_can_broadcast_to_output(arr_in: 'array', arr_out: 'array'):
    if not isinstance(arr_out, array):
        raise TypeError('return arrays must be of modin.numpy.array type.')
    broadcast_ok = arr_in._ndim == arr_out._ndim and arr_in.shape == arr_out.shape or (arr_in._ndim == 2 and arr_out._ndim == 2 and (arr_in.shape[0] == 1) and (arr_in.shape[1] == arr_out.shape[1])) or (arr_in._ndim == 1 and arr_out._ndim == 2 and (arr_in.shape[0] == arr_out.shape[1]) and (arr_out.shape[0] == 1))
    if arr_in._ndim == 2 and arr_out._ndim == 2 and (arr_in.shape[0] == 1) and (arr_in.shape[1] == arr_out.shape[1]) and (arr_in.shape[0] != 1):
        raise NotImplementedError(f'Modin does not currently support broadcasting shape {arr_in.shape} to output operand with shape {arr_out.shape}')
    if not broadcast_ok:
        raise ValueError(f"non-broadcastable output operand with shape {arr_out.shape} doesn't match the broadcast shape {arr_in.shape}")