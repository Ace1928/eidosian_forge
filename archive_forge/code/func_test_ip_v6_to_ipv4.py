from netaddr import IPAddress, IPNetwork
def test_ip_v6_to_ipv4():
    assert IPNetwork('::ffff:192.0.2.1/119').ipv6(ipv4_compatible=True) == IPNetwork('::192.0.2.1/119')
    assert IPNetwork('::ffff:192.0.2.1/119').ipv4() == IPNetwork('192.0.2.1/23')
    assert IPNetwork('::192.0.2.1/119').ipv4() == IPNetwork('192.0.2.1/23')