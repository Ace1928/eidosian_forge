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
def as_known(self, **kwargs):
    """Ensure the categories in this series are known.

        If the categories are known, this is a no-op. If unknown, the
        categories are computed, and a new series with known categories is
        returned.

        Parameters
        ----------
        kwargs
            Keywords to pass on to the call to `compute`.
        """
    if self.known:
        return self._series
    categories = self._property_map('categories').unique().compute(**kwargs)
    return self.set_categories(categories.values)