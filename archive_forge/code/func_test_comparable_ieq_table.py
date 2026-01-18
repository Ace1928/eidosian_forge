from __future__ import print_function, division, absolute_import
from datetime import datetime
from decimal import Decimal
import pytest
from petl.test.helpers import eq_, ieq
from petl.comparison import Comparable
def test_comparable_ieq_table():
    rows = [[u'Bob', 42, 33], [u'Jim', 13, 69], [u'Joe', 86, 17], [u'Ted', 23, 51]]
    ieq(rows, rows)