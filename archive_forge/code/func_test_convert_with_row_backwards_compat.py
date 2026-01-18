from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_convert_with_row_backwards_compat():
    table = (('foo', 'bar'), (' a ', 1), (' b ', 2))
    expect = (('foo', 'bar'), ('a', 1), ('b', 2))
    actual = convert(table, 'foo', 'strip')
    ieq(expect, actual)