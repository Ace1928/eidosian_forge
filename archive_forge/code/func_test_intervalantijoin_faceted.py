from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_intervalantijoin_faceted():
    left = (('fruit', 'begin', 'end'), ('apple', 1, 2), ('apple', 2, 4), ('apple', 2, 5), ('orange', 2, 5), ('orange', 9, 14), ('orange', 19, 140), ('apple', 1, 1), ('apple', 2, 2), ('apple', 4, 4), ('apple', 5, 5), ('orange', 5, 5))
    right = (('type', 'start', 'stop', 'value'), ('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar'), ('orange', 4, 9, 'baz'))
    expect = (('fruit', 'begin', 'end'), ('orange', 9, 14), ('orange', 19, 140), ('apple', 1, 1), ('apple', 2, 2), ('apple', 4, 4), ('apple', 5, 5), ('orange', 5, 5))
    actual = intervalantijoin(left, right, lstart='begin', lstop='end', rstart='start', rstop='stop', lkey='fruit', rkey='type')
    ieq(expect, actual)
    ieq(expect, actual)