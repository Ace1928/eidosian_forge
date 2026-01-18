import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_extend_from_list(self, d):
    d.extend([('set-cookie', '100'), ('set-cookie', '200'), ('set-cookie', '300')])
    assert d['set-cookie'] == '100, 200, 300'