from __future__ import annotations
import pandas as pd
from dask import is_dask_collection
from dask.utils import Dispatch
from_pyarrow_table_dispatch = Dispatch("from_pyarrow_table_dispatch")
def is_categorical_dtype(obj):
    obj = getattr(obj, 'dtype', obj)
    func = is_categorical_dtype_dispatch.dispatch(type(obj))
    return func(obj)