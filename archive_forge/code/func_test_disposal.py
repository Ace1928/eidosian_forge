import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_disposal(self):
    evicted_items = []

    def dispose_func(arg):
        evicted_items.append(arg)
    d = Container(5, dispose_func=dispose_func)
    for i in xrange(5):
        d[i] = i
    assert list(d.keys()) == list(xrange(5))
    assert evicted_items == []
    d[5] = 5
    assert list(d.keys()) == list(xrange(1, 6))
    assert evicted_items == [0]
    del d[1]
    assert evicted_items == [0, 1]
    d.clear()
    assert evicted_items == [0, 1, 2, 3, 4, 5]