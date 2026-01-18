from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_facetintervallookup():
    table = (('type', 'start', 'stop', 'value'), ('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar'), ('orange', 4, 9, 'baz'))
    lkp = facetintervallookup(table, key='type', start='start', stop='stop')
    actual = lkp['apple'].search(0, 1)
    expect = []
    eq_(expect, actual)
    actual = lkp['apple'].search(1, 2)
    expect = [('apple', 1, 4, 'foo')]
    eq_(expect, actual)
    actual = lkp['apple'].search(2, 4)
    expect = [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
    eq_(expect, actual)
    actual = lkp['apple'].search(2, 5)
    expect = [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
    eq_(expect, actual)
    actual = lkp['orange'].search(2, 5)
    expect = [('orange', 4, 9, 'baz')]
    eq_(expect, actual)
    actual = lkp['orange'].search(9, 14)
    expect = []
    eq_(expect, actual)
    actual = lkp['orange'].search(19, 140)
    expect = []
    eq_(expect, actual)
    actual = lkp['apple'].search(0)
    expect = []
    eq_(expect, actual)
    actual = lkp['apple'].search(1)
    expect = [('apple', 1, 4, 'foo')]
    eq_(expect, actual)
    actual = lkp['apple'].search(2)
    expect = [('apple', 1, 4, 'foo')]
    eq_(expect, actual)
    actual = lkp['apple'].search(4)
    expect = [('apple', 3, 7, 'bar')]
    eq_(expect, actual)
    actual = lkp['apple'].search(5)
    expect = [('apple', 3, 7, 'bar')]
    eq_(expect, actual)
    actual = lkp['orange'].search(5)
    expect = [('orange', 4, 9, 'baz')]
    eq_(expect, actual)