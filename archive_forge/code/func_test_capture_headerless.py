from __future__ import absolute_import, print_function, division
import pytest
from petl.compat import next
from petl.errors import ArgumentError
from petl.test.helpers import ieq, eq_
from petl.transform.regex import capture, split, search, searchcomplement, splitdown
from petl.transform.basics import TransformError
def test_capture_headerless():
    table = []
    with pytest.raises(ArgumentError):
        for i in capture(table, 'bar', '(\\w)(\\d)', ('baz', 'qux')):
            pass