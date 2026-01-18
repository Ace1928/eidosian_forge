import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipv6_unicast_address_allocation_info():
    ip = IPNetwork('2001:1200::/23')
    assert ip.info.IPv6[0].allocation == 'Global Unicast'
    assert ip.info.IPv6[0].prefix == '2000::/3'
    assert ip.info.IPv6[0].reference == 'rfc4291'
    assert ip.info.IPv6_unicast[0].prefix == '2001:1200::/23'
    assert ip.info.IPv6_unicast[0].description == 'LACNIC'
    assert ip.info.IPv6_unicast[0].whois == 'whois.lacnic.net'
    assert ip.info.IPv6_unicast[0].status == 'ALLOCATED'