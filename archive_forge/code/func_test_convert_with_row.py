from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_convert_with_row():
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    expect = (('foo', 'bar'), ('a', 'A'), ('b', 'B'))
    actual = convert(table, 'bar', lambda v, row: row.foo.upper(), pass_row=True)
    ieq(expect, actual)