from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_convertall():
    table1 = (('foo', 'bar', 'baz'), ('1', '3', '9'), ('2', '1', '7'))
    table2 = convertall(table1, int)
    expect2 = (('foo', 'bar', 'baz'), (1, 3, 9), (2, 1, 7))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table1 = (('foo', 3, 4), (2, 2, 2))
    table2 = convertall(table1, lambda x: x ** 2)
    expect = (('foo', 3, 4), (4, 4, 4))
    ieq(expect, table2)