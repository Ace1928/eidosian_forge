from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register_lazy('pandas')
def register_pandas():
    import numpy as np
    import pandas as pd
    OBJECT_DTYPES = (object, pd.StringDtype('python'))

    def object_size(*xs):
        if not xs:
            return 0
        ncells = sum((len(x) for x in xs))
        if not ncells:
            return 0
        unique_samples = {}
        for x in xs:
            sample = np.random.choice(x, size=100, replace=True)
            for i in sample.tolist():
                unique_samples[id(i)] = i
        nsamples = 100 * len(xs)
        sample_nbytes = sum((sizeof(i) for i in unique_samples.values()))
        if len(unique_samples) / nsamples > 0.5:
            return int(sample_nbytes * ncells / nsamples)
        else:
            return sample_nbytes

    @sizeof.register(pd.DataFrame)
    def sizeof_pandas_dataframe(df):
        p = sizeof(df.index) + sizeof(df.columns)
        object_cols = []
        prev_dtype = None
        for col in df._series.values():
            if prev_dtype is None or col.dtype != prev_dtype:
                prev_dtype = col.dtype
                p += 1200
            p += col.memory_usage(index=False, deep=False)
            if col.dtype in OBJECT_DTYPES:
                object_cols.append(col._values)
        p += object_size(*object_cols)
        return max(1200, p)

    @sizeof.register(pd.Series)
    def sizeof_pandas_series(s):
        p = 1200 + sizeof(s.index) + s.memory_usage(index=False, deep=False)
        if s.dtype in OBJECT_DTYPES:
            p += object_size(s._values)
        return p

    @sizeof.register(pd.Index)
    def sizeof_pandas_index(i):
        p = 400 + i.memory_usage(deep=False)
        if i.dtype in OBJECT_DTYPES:
            p += object_size(i)
        return p

    @sizeof.register(pd.MultiIndex)
    def sizeof_pandas_multiindex(i):
        return sum((sizeof(l) for l in i.levels)) + sum((c.nbytes for c in i.codes))