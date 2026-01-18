from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
def make_partition(columns: list, dtypes: dict[str, type | str], index, kwargs, state):
    data = {}
    for k, dt in dtypes.items():
        kws = {kk.rsplit('_', 1)[1]: v for kk, v in kwargs.items() if kk.rsplit('_', 1)[0] == k}
        result = make[dt](len(index), state, **kws)
        if k in columns:
            data[k] = result
    df = pd.DataFrame(data, index=index, columns=columns)
    update_dtypes = {k: v for k, v in dtypes.items() if k in columns and (not same_astype(v, df[k].dtype))}
    if update_dtypes:
        kwargs = {} if PANDAS_GE_300 else {'copy': False}
        df = df.astype(update_dtypes, **kwargs)
    return df