import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_0d_recarray_repr(self):
    arr_0d = np.rec.array((1, 2.0, '2003'), dtype='<i4,<f8,<M8[Y]')
    assert_equal(repr(arr_0d), textwrap.dedent("            rec.array((1, 2., '2003'),\n                      dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<M8[Y]')])"))
    record = arr_0d[()]
    assert_equal(repr(record), "(1, 2., '2003')")
    try:
        np.set_printoptions(legacy='1.13')
        assert_equal(repr(record), '(1, 2.0, datetime.date(2003, 1, 1))')
    finally:
        np.set_printoptions(legacy=False)