from __future__ import division, print_function, absolute_import
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xls import fromxls, toxls
from petl.test.helpers import ieq
def test_fromxls():
    filename = _get_test_xls()
    if filename is None:
        return
    tbl = fromxls(filename, 'Sheet1')
    expect = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 2), (u'é', datetime(2012, 1, 1)))
    ieq(expect, tbl)
    ieq(expect, tbl)