from __future__ import annotations
import warnings
import pytest
from dask import utils_test
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import _check_warning
def test__check_warning():

    class MyWarning(Warning):
        pass
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with _check_warning(True, MyWarning, 'foo'):
            warnings.warn('foo', MyWarning)
    with pytest.warns(MyWarning, match='foo'):
        with _check_warning(False, MyWarning, 'foo'):
            warnings.warn('foo', MyWarning)