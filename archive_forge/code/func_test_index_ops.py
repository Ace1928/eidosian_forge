import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('func', [lambda x: x._shallow_copy(x._values), lambda x: x.view(), lambda x: x.take([0, 1]), lambda x: x.repeat([1, 1]), lambda x: x[slice(0, 2)], lambda x: x[[0, 1]], lambda x: x._getitem_slice(slice(0, 2)), lambda x: x.delete([]), lambda x: x.rename('b'), lambda x: x.astype('Int64', copy=False)], ids=['_shallow_copy', 'view', 'take', 'repeat', 'getitem_slice', 'getitem_list', '_getitem_slice', 'delete', 'rename', 'astype'])
def test_index_ops(using_copy_on_write, func, request):
    idx, view_ = index_view()
    expected = idx.copy(deep=True)
    if 'astype' in request.node.callspec.id:
        expected = expected.astype('Int64')
    idx = func(idx)
    view_.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected, check_names=False)