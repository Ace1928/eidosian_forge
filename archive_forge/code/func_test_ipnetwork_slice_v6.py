import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipnetwork_slice_v6():
    ip = IPNetwork('fe80::/10')
    assert ip[0] == IPAddress('fe80::')
    assert ip[-1] == IPAddress('febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff')
    assert ip.size == 332306998946228968225951765070086144
    with pytest.raises(TypeError):
        list(ip[0:5:1])