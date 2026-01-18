from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_head_raises_stop_iteration_for_header_only():
    table1 = (('foo', 'bar', 'baz'),)
    table = iter(head(table1))
    next(table)
    with pytest.raises(StopIteration):
        next(table)