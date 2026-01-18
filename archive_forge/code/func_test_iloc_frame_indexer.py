from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_frame_indexer(self):
    df = DataFrame({'a': [1, 2, 3]})
    indexer = DataFrame({'a': [True, False, True]})
    msg = 'DataFrame indexer for .iloc is not supported. Consider using .loc'
    with pytest.raises(TypeError, match=msg):
        df.iloc[indexer] = 1
    msg = 'DataFrame indexer is not allowed for .iloc\nConsider using .loc for automatic alignment.'
    with pytest.raises(IndexError, match=msg):
        df.iloc[indexer]