from __future__ import division, print_function, absolute_import
import pytest
import petl as etl
from petl.test.helpers import ieq
from petl.io.pandas import todataframe, fromdataframe
def test_headerless():
    tbl = []
    expect = pd.DataFrame()
    actual = todataframe(tbl)
    assert expect.equals(actual)