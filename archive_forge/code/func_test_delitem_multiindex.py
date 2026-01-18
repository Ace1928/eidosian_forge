import re
import numpy as np
import pytest
from pandas import (
def test_delitem_multiindex(self):
    midx = MultiIndex.from_product([['A', 'B'], [1, 2]])
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=midx)
    assert len(df.columns) == 4
    assert ('A',) in df.columns
    assert 'A' in df.columns
    result = df['A']
    assert isinstance(result, DataFrame)
    del df['A']
    assert len(df.columns) == 2
    assert ('A',) not in df.columns
    with pytest.raises(KeyError, match=re.escape("('A',)")):
        del df['A',]
    assert 'A' not in df.columns
    with pytest.raises(KeyError, match=re.escape("('A',)")):
        del df['A']