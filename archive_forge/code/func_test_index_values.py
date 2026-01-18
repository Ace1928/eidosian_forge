import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_index_values(using_copy_on_write):
    idx = Index([1, 2, 3])
    result = idx.values
    if using_copy_on_write:
        assert result.flags.writeable is False
    else:
        assert result.flags.writeable is True