from netaddr import IPNetwork
def test_is_multicast():
    assert IPNetwork('239.192.0.1/24').is_multicast()
    assert IPNetwork('ff00::/8').is_multicast()