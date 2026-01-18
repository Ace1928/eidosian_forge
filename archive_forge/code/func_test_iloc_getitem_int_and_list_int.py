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
@pytest.mark.parametrize('key', [2, -1, [0, 1, 2]])
@pytest.mark.parametrize('kind', ['series', 'frame'])
@pytest.mark.parametrize('col', ['labels', 'mixed', 'ts', 'floats', 'empty'])
def test_iloc_getitem_int_and_list_int(self, key, kind, col, request):
    obj = request.getfixturevalue(f'{kind}_{col}')
    check_indexing_smoketest_or_raises(obj, 'iloc', key, fails=IndexError)