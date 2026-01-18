import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def pandas_feature_info(data: DataFrame, meta: Optional[str], feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], enable_categorical: bool) -> Tuple[Optional[FeatureNames], Optional[FeatureTypes]]:
    """Handle feature info for pandas dataframe."""
    import pandas as pd
    if feature_names is None and meta is None:
        if isinstance(data.columns, pd.MultiIndex):
            feature_names = [' '.join([str(x) for x in i]) for i in data.columns]
        elif isinstance(data.columns, (pd.Index, pd.RangeIndex)):
            feature_names = list(map(str, data.columns))
        else:
            feature_names = data.columns.format()
    if feature_types is None and meta is None:
        feature_types = []
        for dtype in data.dtypes:
            if is_pd_sparse_dtype(dtype):
                feature_types.append(_pandas_dtype_mapper[dtype.subtype.name])
            elif (is_pd_cat_dtype(dtype) or is_pa_ext_categorical_dtype(dtype)) and enable_categorical:
                feature_types.append(CAT_T)
            else:
                feature_types.append(_pandas_dtype_mapper[dtype.name])
    return (feature_names, feature_types)