import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_get_indexer_consistency(self, index):
    if index._index_as_unique:
        indexer = index.get_indexer(index[0:2])
        assert isinstance(indexer, np.ndarray)
        assert indexer.dtype == np.intp
    else:
        msg = 'Reindexing only valid with uniquely valued Index objects'
        with pytest.raises(InvalidIndexError, match=msg):
            index.get_indexer(index[0:2])
    indexer, _ = index.get_indexer_non_unique(index[0:2])
    assert isinstance(indexer, np.ndarray)
    assert indexer.dtype == np.intp