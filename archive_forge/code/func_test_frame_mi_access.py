import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
@pytest.mark.parametrize('indexer', [lambda df: df['A', 'A1'], lambda df: df.loc[:, ('A', 'A1')]])
def test_frame_mi_access(dataframe_with_duplicate_index, indexer):
    df = dataframe_with_duplicate_index
    index = Index(['h1', 'h3', 'h5'])
    columns = MultiIndex.from_tuples([('A', 'A1')], names=['main', 'sub'])
    expected = DataFrame([['a', 1, 1]], index=columns, columns=index).T
    result = indexer(df)
    tm.assert_frame_equal(result, expected)