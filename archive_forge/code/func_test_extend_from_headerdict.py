import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_extend_from_headerdict(self, d):
    h = HTTPHeaderDict(Cookie='foo', e='foofoo')
    d.extend(h)
    assert d['cookie'] == 'foo, bar, foo'
    assert d['e'] == 'foofoo'
    assert len(d) == 2