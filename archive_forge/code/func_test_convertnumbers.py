from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_convertnumbers():
    table1 = (('foo', 'bar', 'baz', 'quux'), ('1', '3.0', '9+3j', 'aaa'), ('2', '1.3', '7+2j', None))
    table2 = convertnumbers(table1)
    expect2 = (('foo', 'bar', 'baz', 'quux'), (1, 3.0, 9 + 3j, 'aaa'), (2, 1.3, 7 + 2j, None))
    ieq(expect2, table2)
    ieq(expect2, table2)