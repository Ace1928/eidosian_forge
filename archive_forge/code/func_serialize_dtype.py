import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def serialize_dtype(self, dtype):
    """
        Serialize data type to a dictionary.

        Parameters
        ----------
        dtype : dtype
            Data type to serialize.

        Returns
        -------
        dict
            Serialized data type.
        """
    _warn_if_unsigned(dtype)
    try:
        type_info = {'type': self._DTYPE_STRINGS[dtype.name], 'nullable': True}
        if is_datetime64_dtype(dtype):
            unit = np.datetime_data(dtype)[0]
            type_info['precision'] = self._TIMESTAMP_PRECISION[unit]
        return type_info
    except KeyError:
        raise TypeError(f'Unsupported dtype: {dtype}')