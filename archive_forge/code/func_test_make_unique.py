import numpy as np
import pytest
from pandas.core.apply import (
@pytest.mark.parametrize('order, expected_reorder', [([('height', '<lambda>'), ('height', 'max'), ('weight', 'max'), ('height', '<lambda>'), ('weight', '<lambda>')], [('height', '<lambda>_0'), ('height', 'max'), ('weight', 'max'), ('height', '<lambda>_1'), ('weight', '<lambda>')]), ([('col2', 'min'), ('col1', '<lambda>'), ('col1', '<lambda>'), ('col1', '<lambda>')], [('col2', 'min'), ('col1', '<lambda>_0'), ('col1', '<lambda>_1'), ('col1', '<lambda>_2')]), ([('col', '<lambda>'), ('col', '<lambda>'), ('col', '<lambda>')], [('col', '<lambda>_0'), ('col', '<lambda>_1'), ('col', '<lambda>_2')])])
def test_make_unique(order, expected_reorder):
    result = _make_unique_kwarg_list(order)
    assert result == expected_reorder