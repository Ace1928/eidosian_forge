import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_join_level_corner_case(idx):
    index = Index(['three', 'one', 'two'])
    result = index.join(idx, level='second')
    assert isinstance(result, MultiIndex)
    with pytest.raises(TypeError, match='Join.*MultiIndex.*ambiguous'):
        idx.join(idx, level=1)