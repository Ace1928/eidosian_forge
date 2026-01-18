import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_create_from_list(self):
    headers = [('ab', 'A'), ('cd', 'B'), ('cookie', 'C'), ('cookie', 'D'), ('cookie', 'E')]
    h = HTTPHeaderDict(headers)
    assert len(h) == 3
    assert 'ab' in h
    clist = h.getlist('cookie')
    assert len(clist) == 3
    assert clist[0] == 'C'
    assert clist[-1] == 'E'