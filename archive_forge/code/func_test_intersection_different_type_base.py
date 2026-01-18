from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('klass', [np.array, Series, list])
def test_intersection_different_type_base(self, klass, sort):
    index = Index([0, 'a', 1, 'b', 2, 'c'])
    first = index[:5]
    second = index[:3]
    result = first.intersection(klass(second.values), sort=sort)
    assert equal_contents(result, second)