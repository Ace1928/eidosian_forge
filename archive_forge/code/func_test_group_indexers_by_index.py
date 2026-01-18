from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
def test_group_indexers_by_index(self) -> None:
    mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('one', 'two'))
    data = DataArray(np.zeros((4, 2, 2)), coords={'x': mindex, 'y': [1, 2]}, dims=('x', 'y', 'z'))
    data.coords['y2'] = ('y', [2.0, 3.0])
    grouped_indexers = indexing.group_indexers_by_index(data, {'z': 0, 'one': 'a', 'two': 1, 'y': 0}, {})
    for idx, indexers in grouped_indexers:
        if idx is None:
            assert indexers == {'z': 0}
        elif idx.equals(data.xindexes['x']):
            assert indexers == {'one': 'a', 'two': 1}
        elif idx.equals(data.xindexes['y']):
            assert indexers == {'y': 0}
    assert len(grouped_indexers) == 3
    with pytest.raises(KeyError, match="no index found for coordinate 'y2'"):
        indexing.group_indexers_by_index(data, {'y2': 2.0}, {})
    with pytest.raises(KeyError, match="'w' is not a valid dimension or coordinate"):
        indexing.group_indexers_by_index(data, {'w': 'a'}, {})
    with pytest.raises(ValueError, match='cannot supply.*'):
        indexing.group_indexers_by_index(data, {'z': 1}, {'method': 'nearest'})