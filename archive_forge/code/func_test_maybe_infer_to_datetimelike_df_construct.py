import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('data,exp_size', [([[NaT, 'a', 'b', 0], [NaT, 'b', 'c', 1]], 8), ([[NaT, 'a', 0], [NaT, 'b', 1]], 6)])
def test_maybe_infer_to_datetimelike_df_construct(data, exp_size):
    result = DataFrame(np.array(data))
    assert result.size == exp_size