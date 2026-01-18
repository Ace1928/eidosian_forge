import warnings
import pytest
from urllib3.connection import HTTPConnection
from urllib3.packages.six.moves import http_cookiejar, urllib
from urllib3.response import HTTPResponse
def test_connection_strict(self):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        HTTPConnection('localhost', 12345, strict=True)
        if w:
            pytest.fail('HTTPConnection raised warning on strict=True: %r' % w[0].message)