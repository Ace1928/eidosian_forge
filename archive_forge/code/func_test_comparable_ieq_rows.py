from __future__ import print_function, division, absolute_import
from datetime import datetime
from decimal import Decimal
import pytest
from petl.test.helpers import eq_, ieq
from petl.comparison import Comparable
def test_comparable_ieq_rows():
    rows = [['a', 'b', 'c'], [1, 2]]
    ieq(rows, rows)