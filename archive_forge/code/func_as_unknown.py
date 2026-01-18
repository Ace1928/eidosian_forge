from __future__ import annotations
from collections import defaultdict
from numbers import Integral
import pandas as pd
from pandas.api.types import is_scalar
from tlz import partition_all
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.accessor import Accessor
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
def as_unknown(self):
    """Ensure the categories in this series are unknown"""
    if not self.known:
        return self._series
    out = self._series.copy()
    out._meta = clear_known_categories(out._meta)
    return out