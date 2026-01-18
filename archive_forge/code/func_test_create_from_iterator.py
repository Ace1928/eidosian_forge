import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_create_from_iterator(self):
    teststr = 'urllib3ontherocks'
    h = HTTPHeaderDict(((c, c * 5) for c in teststr))
    assert len(h) == len(set(teststr))