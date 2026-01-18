from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_cutout():
    table = (('foo', 'bar', 'baz'), ('A', 1, 2), ('B', '2', '3.4'), (u'B', u'3', u'7.8', True), ('D', 'xyz', 9.0), ('E', None))
    cut1 = cutout(table, 'bar', 'baz')
    expectation = (('foo',), ('A',), ('B',), (u'B',), ('D',), ('E',))
    ieq(expectation, cut1)
    cut2 = cutout(table, 'bar')
    expectation = (('foo', 'baz'), ('A', 2), ('B', '3.4'), (u'B', u'7.8'), ('D', 9.0), ('E', None))
    ieq(expectation, cut2)
    cut3 = cutout(table, 1)
    expectation = (('foo', 'baz'), ('A', 2), ('B', '3.4'), (u'B', u'7.8'), ('D', 9.0), ('E', None))
    ieq(expectation, cut3)