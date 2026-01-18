import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_add_comma_separated_multiheader(self, d):
    d.add('bar', 'foo')
    d.add('BAR', 'bar')
    d.add('Bar', 'asdf')
    assert d.getlist('bar') == ['foo', 'bar', 'asdf']
    assert d['bar'] == 'foo, bar, asdf'