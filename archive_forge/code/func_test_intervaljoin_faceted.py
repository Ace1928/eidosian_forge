from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_intervaljoin_faceted():
    left = (('fruit', 'begin', 'end'), ('apple', 1, 2), ('apple', 2, 4), ('apple', 2, 5), ('orange', 2, 5), ('orange', 9, 14), ('orange', 19, 140), ('apple', 1, 1), ('apple', 2, 2), ('apple', 4, 4), ('apple', 5, 5), ('orange', 5, 5))
    right = (('type', 'start', 'stop', 'value'), ('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar'), ('orange', 4, 9, 'baz'))
    expect = (('fruit', 'begin', 'end', 'type', 'start', 'stop', 'value'), ('apple', 1, 2, 'apple', 1, 4, 'foo'), ('apple', 2, 4, 'apple', 1, 4, 'foo'), ('apple', 2, 4, 'apple', 3, 7, 'bar'), ('apple', 2, 5, 'apple', 1, 4, 'foo'), ('apple', 2, 5, 'apple', 3, 7, 'bar'), ('orange', 2, 5, 'orange', 4, 9, 'baz'))
    actual = intervaljoin(left, right, lstart='begin', lstop='end', rstart='start', rstop='stop', lkey='fruit', rkey='type')
    ieq(expect, actual)
    ieq(expect, actual)