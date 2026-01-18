from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def test_complement_seqtypes():
    ta = [['a', 'b'], ['A', 1], ['B', 2]]
    tb = [('a', 'b'), ('A', 1), ('B', 2)]
    expectation = (('a', 'b'),)
    actual = complement(ta, tb, presorted=True)
    ieq(expectation, actual)