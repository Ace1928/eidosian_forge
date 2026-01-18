from __future__ import print_function, division, absolute_import
from datetime import datetime
from decimal import Decimal
import pytest
from petl.test.helpers import eq_, ieq
from petl.comparison import Comparable
def test_comparable_nested():
    d = [[3], [1], [2]]
    a = sorted(d, key=Comparable)
    e = [[1], [2], [3]]
    eq_(e, a)
    d = [(3,), (1,), (2,)]
    a = sorted(d, key=Comparable)
    e = [(1,), (2,), (3,)]
    eq_(e, a)
    d = [3, 1, [2]]
    a = sorted(d, key=Comparable)
    e = [1, 3, [2]]
    eq_(e, a)
    d = [[3], [None], [2]]
    a = sorted(d, key=Comparable)
    e = [[None], [2], [3]]
    eq_(e, a)
    d = [[3], [1], (2,)]
    a = sorted(d, key=Comparable)
    e = [[1], (2,), [3]]
    eq_(e, a)
    d = [[3, 2], [3, 1], [2]]
    a = sorted(d, key=Comparable)
    e = [[2], [3, 1], [3, 2]]
    eq_(e, a)
    dt = datetime.now().replace
    d = [dt(hour=12), None, (dt(hour=3), 'b'), True, [b'aa', False], (b'aa', -1), 3.4]
    a = sorted(d, key=Comparable)
    e = [None, True, 3.4, dt(hour=12), (dt(hour=3), 'b'), (b'aa', -1), [b'aa', False]]
    eq_(e, a)