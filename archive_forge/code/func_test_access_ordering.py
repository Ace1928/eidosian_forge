import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_access_ordering(self):
    d = Container(5)
    for i in xrange(10):
        d[i] = True
    assert list(d.keys()) == [5, 6, 7, 8, 9]
    new_order = [7, 8, 6, 9, 5]
    for k in new_order:
        d[k]
    assert list(d.keys()) == new_order