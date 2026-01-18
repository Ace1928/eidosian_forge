import pytest
from pandas import (
import pandas._testing as tm
def test_as_index_false_unsupported(self, numba_supported_reductions):
    func, kwargs = numba_supported_reductions
    df = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
    gb = df.groupby('a', as_index=False)
    with pytest.raises(NotImplementedError, match='as_index=False'):
        getattr(gb, func)(engine='numba', **kwargs)