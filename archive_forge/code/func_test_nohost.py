import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
@pytest.mark.parametrize('url', ['http://@', None])
def test_nohost(self, url):
    p = PoolManager(5)
    with pytest.raises(LocationValueError):
        p.connection_from_url(url=url)