from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def test_hashcomplement():
    _test_complement(hashcomplement)