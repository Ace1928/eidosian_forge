from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_extendheader():
    table1 = (('foo',), ('a', 1, True), ('b', 2, False))
    table2 = extendheader(table1, ['bar', 'baz'])
    expect2 = (('foo', 'bar', 'baz'), ('a', 1, True), ('b', 2, False))
    ieq(expect2, table2)
    ieq(expect2, table2)