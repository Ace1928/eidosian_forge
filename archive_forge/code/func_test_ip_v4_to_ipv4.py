from netaddr import IPAddress, IPNetwork
def test_ip_v4_to_ipv4():
    assert IPAddress('192.0.2.15').ipv4() == IPAddress('192.0.2.15')