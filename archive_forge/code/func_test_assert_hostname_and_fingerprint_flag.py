import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_assert_hostname_and_fingerprint_flag(self):
    """Assert that pool manager can accept hostname and fingerprint flags."""
    fingerprint = '92:81:FE:85:F7:0C:26:60:EC:D6:B3:BF:93:CF:F9:71:CC:07:7D:0A'
    p = PoolManager(assert_hostname=True, assert_fingerprint=fingerprint)
    pool = p.connection_from_url('https://example.com/')
    assert 1 == len(p.pools)
    assert pool.assert_hostname
    assert fingerprint == pool.assert_fingerprint