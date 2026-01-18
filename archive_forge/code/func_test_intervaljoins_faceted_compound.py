from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_intervaljoins_faceted_compound():
    left = (('fruit', 'sort', 'begin', 'end'), ('apple', 'cox', 1, 2), ('apple', 'fuji', 2, 4))
    right = (('type', 'variety', 'start', 'stop', 'value'), ('apple', 'cox', 1, 4, 'foo'), ('apple', 'fuji', 3, 7, 'bar'), ('orange', 'mandarin', 4, 9, 'baz'))
    expect = (('fruit', 'sort', 'begin', 'end', 'type', 'variety', 'start', 'stop', 'value'), ('apple', 'cox', 1, 2, 'apple', 'cox', 1, 4, 'foo'), ('apple', 'fuji', 2, 4, 'apple', 'fuji', 3, 7, 'bar'))
    actual = intervaljoin(left, right, lstart='begin', lstop='end', rstart='start', rstop='stop', lkey=('fruit', 'sort'), rkey=('type', 'variety'))
    ieq(expect, actual)
    ieq(expect, actual)
    actual = intervalleftjoin(left, right, lstart='begin', lstop='end', rstart='start', rstop='stop', lkey=('fruit', 'sort'), rkey=('type', 'variety'))
    ieq(expect, actual)
    ieq(expect, actual)