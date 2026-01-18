import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_converter_with_structured_dtype():
    txt = StringIO('1.5,2.5,Abc\n3.0,4.0,dEf\n5.5,6.0,ghI\n')
    dt = np.dtype([('m', np.int32), ('r', np.float32), ('code', 'U8')])
    conv = {0: lambda s: int(10 * float(s)), -1: lambda s: s.upper()}
    res = np.loadtxt(txt, dtype=dt, delimiter=',', converters=conv)
    expected = np.array([(15, 2.5, 'ABC'), (30, 4.0, 'DEF'), (55, 6.0, 'GHI')], dtype=dt)
    assert_equal(res, expected)