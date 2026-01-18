import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_dict_conversion(self, d):
    hdict = {'Content-Length': '0', 'Content-type': 'text/plain', 'Server': 'TornadoServer/1.2.3'}
    h = dict(HTTPHeaderDict(hdict).items())
    assert hdict == h
    assert hdict == dict(HTTPHeaderDict(hdict))