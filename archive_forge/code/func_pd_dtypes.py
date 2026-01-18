import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
def pd_dtypes() -> Generator:
    """Enumerate all supported pandas extension types."""
    import pandas as pd
    dtypes = [pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype(), pd.Int8Dtype(), pd.Int16Dtype(), pd.Int32Dtype(), pd.Int64Dtype()]
    Null: Union[float, None, Any] = np.nan
    orig = pd.DataFrame({'f0': [1, 2, Null, 3], 'f1': [4, 3, Null, 1]}, dtype=np.float32)
    for Null in (np.nan, None, pd.NA):
        for dtype in dtypes:
            df = pd.DataFrame({'f0': [1, 2, Null, 3], 'f1': [4, 3, Null, 1]}, dtype=dtype)
            yield (orig, df)
    Null = np.nan
    dtypes = [pd.Float32Dtype(), pd.Float64Dtype()]
    orig = pd.DataFrame({'f0': [1.0, 2.0, Null, 3.0], 'f1': [3.0, 2.0, Null, 1.0]}, dtype=np.float32)
    for Null in (np.nan, None, pd.NA):
        for dtype in dtypes:
            df = pd.DataFrame({'f0': [1.0, 2.0, Null, 3.0], 'f1': [3.0, 2.0, Null, 1.0]}, dtype=dtype)
            yield (orig, df)
            ser_orig = orig['f0']
            ser = df['f0']
            assert isinstance(ser, pd.Series)
            assert isinstance(ser_orig, pd.Series)
            yield (ser_orig, ser)
    orig = orig.astype('category')
    for Null in (np.nan, None, pd.NA):
        df = pd.DataFrame({'f0': [1.0, 2.0, Null, 3.0], 'f1': [3.0, 2.0, Null, 1.0]}, dtype=pd.CategoricalDtype())
        yield (orig, df)
    for Null in [None, pd.NA]:
        data = {'f0': [True, False, Null, True], 'f1': [False, True, Null, True]}
        orig = pd.DataFrame(data, dtype=np.bool_ if Null is None else pd.BooleanDtype())
        df = pd.DataFrame(data, dtype=pd.BooleanDtype())
        yield (orig, df)