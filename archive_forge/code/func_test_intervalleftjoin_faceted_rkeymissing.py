from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_intervalleftjoin_faceted_rkeymissing():
    left = (('fruit', 'begin', 'end'), ('apple', 1, 2), ('orange', 5, 5))
    right = (('type', 'start', 'stop', 'value'), ('apple', 1, 4, 'foo'))
    expect = (('fruit', 'begin', 'end', 'type', 'start', 'stop', 'value'), ('apple', 1, 2, 'apple', 1, 4, 'foo'), ('orange', 5, 5, None, None, None, None))
    actual = intervalleftjoin(left, right, lstart='begin', lstop='end', rstart='start', rstop='stop', lkey='fruit', rkey='type')
    ieq(expect, actual)
    ieq(expect, actual)