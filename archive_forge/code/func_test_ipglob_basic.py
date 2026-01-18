from netaddr import (
def test_ipglob_basic():
    assert IPGlob('192.0.2.*') == IPNetwork('192.0.2.0/24')