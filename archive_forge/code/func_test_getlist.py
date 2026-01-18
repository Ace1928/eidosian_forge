import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_getlist(self, d):
    assert d.getlist('cookie') == ['foo', 'bar']
    assert d.getlist('Cookie') == ['foo', 'bar']
    assert d.getlist('b') == []
    d.add('b', 'asdf')
    assert d.getlist('b') == ['asdf']