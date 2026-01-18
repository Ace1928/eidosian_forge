from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
@pytest.mark.parametrize('mapping', ['mapping', 'dict'])
def test_concat_mapping(self, mapping, non_dict_mapping_subclass):
    constructor = dict if mapping == 'dict' else non_dict_mapping_subclass
    frames = constructor({'foo': DataFrame(np.random.default_rng(2).standard_normal((4, 3))), 'bar': DataFrame(np.random.default_rng(2).standard_normal((4, 3))), 'baz': DataFrame(np.random.default_rng(2).standard_normal((4, 3))), 'qux': DataFrame(np.random.default_rng(2).standard_normal((4, 3)))})
    sorted_keys = list(frames.keys())
    result = concat(frames)
    expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys)
    tm.assert_frame_equal(result, expected)
    result = concat(frames, axis=1)
    expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys, axis=1)
    tm.assert_frame_equal(result, expected)
    keys = ['baz', 'foo', 'bar']
    result = concat(frames, keys=keys)
    expected = concat([frames[k] for k in keys], keys=keys)
    tm.assert_frame_equal(result, expected)