from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_metadata_propagation(self, frame_or_series):
    o = construct(frame_or_series, shape=3)
    o.name = 'foo'
    o2 = construct(frame_or_series, shape=3)
    o2.name = 'bar'
    for op in ['__add__', '__sub__', '__truediv__', '__mul__']:
        result = getattr(o, op)(1)
        tm.assert_metadata_equivalent(o, result)
    for op in ['__add__', '__sub__', '__truediv__', '__mul__']:
        result = getattr(o, op)(o)
        tm.assert_metadata_equivalent(o, result)
    for op in ['__eq__', '__le__', '__ge__']:
        v1 = getattr(o, op)(o)
        tm.assert_metadata_equivalent(o, v1)
        tm.assert_metadata_equivalent(o, v1 & v1)
        tm.assert_metadata_equivalent(o, v1 | v1)
    result = o.combine_first(o2)
    tm.assert_metadata_equivalent(o, result)
    result = o + o2
    tm.assert_metadata_equivalent(result)
    for op in ['__eq__', '__le__', '__ge__']:
        v1 = getattr(o, op)(o)
        v2 = getattr(o, op)(o2)
        tm.assert_metadata_equivalent(v2)
        tm.assert_metadata_equivalent(v1 & v2)
        tm.assert_metadata_equivalent(v1 | v2)