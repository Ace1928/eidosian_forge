from __future__ import absolute_import, print_function, division
import pytest
from petl.compat import next
from petl.errors import ArgumentError
from petl.test.helpers import ieq, eq_
from petl.transform.regex import capture, split, search, searchcomplement, splitdown
from petl.transform.basics import TransformError
def test_capture_nonmatching():
    table = (('id', 'variable', 'value'), ('1', 'A1', '12'), ('2', 'A2', '15'), ('3', 'B1', '18'), ('4', 'C12', '19'))
    expectation = (('id', 'value', 'treat', 'time'), ('1', '12', 'A', '1'), ('2', '15', 'A', '2'), ('3', '18', 'B', '1'))
    result = capture(table, 'variable', '([A-B])(\\d+)', ('treat', 'time'))
    it = iter(result)
    eq_(expectation[0], next(it))
    eq_(expectation[1], next(it))
    eq_(expectation[2], next(it))
    eq_(expectation[3], next(it))
    try:
        next(it)
    except TransformError:
        pass
    else:
        assert False, 'expected exception'
    result = capture(table, 'variable', '([A-B])(\\d+)', newfields=('treat', 'time'), fill=['', 0])
    it = iter(result)
    eq_(expectation[0], next(it))
    eq_(expectation[1], next(it))
    eq_(expectation[2], next(it))
    eq_(expectation[3], next(it))
    eq_(('4', '19', '', 0), next(it))