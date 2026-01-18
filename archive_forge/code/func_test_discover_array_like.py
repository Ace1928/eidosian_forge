from collections import OrderedDict
from itertools import starmap
from types import MappingProxyType
from warnings import catch_warnings, simplefilter
import numpy as np
import pytest
from datashader.datashape.discovery import (
from datashader.datashape.coretypes import (
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape import dshape
from datetime import date, time, datetime, timedelta
def test_discover_array_like():

    class MyArray:

        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
    with catch_warnings(record=True) as wl:
        simplefilter('always')
        assert discover(MyArray((4, 3), 'f4')) == dshape('4 * 3 * float32')
    assert len(wl) == 1
    assert issubclass(wl[0].category, DeprecationWarning)
    assert 'MyArray' in str(wl[0].message)