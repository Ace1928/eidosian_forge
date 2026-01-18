import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_many_urls(self):
    urls = ['http://localhost:8081/foo', 'http://www.google.com/mail', 'http://localhost:8081/bar', 'https://www.google.com/', 'https://www.google.com/mail', 'http://yahoo.com', 'http://bing.com', 'http://yahoo.com/']
    connections = set()
    p = PoolManager(10)
    for url in urls:
        conn = p.connection_from_url(url)
        connections.add(conn)
    assert len(connections) == 5