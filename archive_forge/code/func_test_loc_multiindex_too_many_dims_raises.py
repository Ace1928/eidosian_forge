import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_multiindex_too_many_dims_raises(self):
    s = Series(range(8), index=MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']]))
    with pytest.raises(KeyError, match="^\\('a', 'b'\\)$"):
        s.loc['a', 'b']
    with pytest.raises(KeyError, match="^\\('a', 'd', 'g'\\)$"):
        s.loc['a', 'd', 'g']
    with pytest.raises(IndexingError, match='Too many indexers'):
        s.loc['a', 'd', 'g', 'j']