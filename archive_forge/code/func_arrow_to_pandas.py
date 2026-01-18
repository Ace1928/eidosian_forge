from __future__ import annotations
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from pandas.core.dtypes.common import _get_dtype, is_string_dtype
from pyarrow.types import is_dictionary
from modin.pandas.indexing import is_range_like
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def arrow_to_pandas(at: pa.Table, dtypes: Optional[Union[ModinDtypes, pandas.Series]]=None) -> pandas.DataFrame:
    """
    Convert the specified arrow table to pandas.

    Parameters
    ----------
    at : pyarrow.Table
        The table to convert.
    dtypes : Union[ModinDtypes, pandas.Series], optional
        Dtypes are used to correctly map PyArrow types to pandas.

    Returns
    -------
    pandas.DataFrame
    """

    def mapper(at):
        if is_dictionary(at) and isinstance(at.value_type, ArrowIntervalType):
            return _CategoricalDtypeMapper
        elif dtypes is not None and any((isinstance(dtype, pandas.core.dtypes.dtypes.ArrowDtype) for dtype in dtypes)):
            dtype_mapping = {pa.int8(): pandas.ArrowDtype(pa.int8()), pa.int16(): pandas.ArrowDtype(pa.int16()), pa.int32(): pandas.ArrowDtype(pa.int32()), pa.int64(): pandas.ArrowDtype(pa.int64()), pa.uint8(): pandas.ArrowDtype(pa.uint8()), pa.uint16(): pandas.ArrowDtype(pa.uint16()), pa.uint32(): pandas.ArrowDtype(pa.uint32()), pa.uint64(): pandas.ArrowDtype(pa.uint64()), pa.bool_(): pandas.ArrowDtype(pa.bool_()), pa.float32(): pandas.ArrowDtype(pa.float32()), pa.float64(): pandas.ArrowDtype(pa.float64()), pa.string(): pandas.ArrowDtype(pa.string())}
            return dtype_mapping.get(at, None)
        return None
    df = at.to_pandas(types_mapper=mapper)
    dtype = {}
    for idx, _type in enumerate(at.schema.types):
        if isinstance(_type, pa.lib.TimestampType) and _type.unit != 'ns':
            dtype[at.schema.names[idx]] = f'datetime64[{_type.unit}]'
    if dtype:
        df = df.astype(dtype)
    return df