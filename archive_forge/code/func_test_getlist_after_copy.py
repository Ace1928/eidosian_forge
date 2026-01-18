import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_getlist_after_copy(self, d):
    assert d.getlist('cookie') == HTTPHeaderDict(d).getlist('cookie')