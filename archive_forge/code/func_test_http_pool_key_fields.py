import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_http_pool_key_fields(self):
    """Assert the HTTPPoolKey fields are honored when selecting a pool."""
    connection_pool_kw = {'timeout': timeout.Timeout(3.14), 'retries': retry.Retry(total=6, connect=2), 'block': True, 'strict': True, 'source_address': '127.0.0.1'}
    p = PoolManager()
    conn_pools = [p.connection_from_url('http://example.com/'), p.connection_from_url('http://example.com:8000/'), p.connection_from_url('http://other.example.com/')]
    for key, value in connection_pool_kw.items():
        p.connection_pool_kw[key] = value
        conn_pools.append(p.connection_from_url('http://example.com/'))
    assert all((x is not y for i, x in enumerate(conn_pools) for j, y in enumerate(conn_pools) if i != j))
    assert all((isinstance(key, PoolKey) for key in p.pools.keys()))