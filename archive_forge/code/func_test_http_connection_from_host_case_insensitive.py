import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_http_connection_from_host_case_insensitive(self):
    """Assert scheme case is ignored when getting the https key class."""
    p = PoolManager()
    pool = p.connection_from_host('example.com', scheme='http')
    other_pool = p.connection_from_host('EXAMPLE.COM', scheme='HTTP')
    assert 1 == len(p.pools)
    assert pool is other_pool
    assert all((isinstance(key, PoolKey) for key in p.pools.keys()))