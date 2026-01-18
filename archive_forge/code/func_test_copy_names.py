import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_copy_names():
    multi_idx = MultiIndex.from_tuples([(1, 2), (3, 4)], names=['MyName1', 'MyName2'])
    multi_idx1 = multi_idx.copy()
    assert multi_idx.equals(multi_idx1)
    assert multi_idx.names == ['MyName1', 'MyName2']
    assert multi_idx1.names == ['MyName1', 'MyName2']
    multi_idx2 = multi_idx.copy(names=['NewName1', 'NewName2'])
    assert multi_idx.equals(multi_idx2)
    assert multi_idx.names == ['MyName1', 'MyName2']
    assert multi_idx2.names == ['NewName1', 'NewName2']
    multi_idx3 = multi_idx.copy(name=['NewName1', 'NewName2'])
    assert multi_idx.equals(multi_idx3)
    assert multi_idx.names == ['MyName1', 'MyName2']
    assert multi_idx3.names == ['NewName1', 'NewName2']
    with pytest.raises(ValueError, match='Length of new names must be 2, got 1'):
        multi_idx.copy(names=['mario'])
    with pytest.raises(TypeError, match='MultiIndex.name must be a hashable type'):
        multi_idx.copy(names=[['mario'], ['luigi']])