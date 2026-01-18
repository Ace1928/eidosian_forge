import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('has_keys', [True, False])
@pytest.mark.parametrize('has_getitem', [True, False])
@pytest.mark.parametrize('has_contains', [True, False])
def test_is_dict_like_duck_type(has_keys, has_getitem, has_contains):

    class DictLike:

        def __init__(self, d) -> None:
            self.d = d
        if has_keys:

            def keys(self):
                return self.d.keys()
        if has_getitem:

            def __getitem__(self, key):
                return self.d.__getitem__(key)
        if has_contains:

            def __contains__(self, key) -> bool:
                return self.d.__contains__(key)
    d = DictLike({1: 2})
    result = inference.is_dict_like(d)
    expected = has_keys and has_getitem and has_contains
    assert result is expected