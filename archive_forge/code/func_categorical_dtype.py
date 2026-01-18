from __future__ import annotations
import pandas as pd
from dask import is_dask_collection
from dask.utils import Dispatch
from_pyarrow_table_dispatch = Dispatch("from_pyarrow_table_dispatch")
def categorical_dtype(meta, categories=None, ordered=False):
    func = categorical_dtype_dispatch.dispatch(type(meta))
    return func(categories=categories, ordered=ordered)