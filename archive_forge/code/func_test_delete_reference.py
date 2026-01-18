import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_delete_reference(using_copy_on_write):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), columns=['a', 'b', 'c'])
    x = df[:]
    del df['b']
    if using_copy_on_write:
        assert df._mgr.blocks[0].refs.has_reference()
        assert df._mgr.blocks[1].refs.has_reference()
        assert x._mgr.blocks[0].refs.has_reference()