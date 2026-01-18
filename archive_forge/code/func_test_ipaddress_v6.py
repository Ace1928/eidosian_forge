import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipaddress_v6():
    ip = IPAddress('fe80::dead:beef')
    assert ip.version == 6
    assert repr(ip) == "IPAddress('fe80::dead:beef')"
    assert str(ip) == 'fe80::dead:beef'
    assert ip.format() == 'fe80::dead:beef'
    assert int(ip) == 338288524927261089654018896845083623151
    assert hex(ip) == '0xfe8000000000000000000000deadbeef'
    assert bytes(ip) == b'\xfe\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xde\xad\xbe\xef'
    assert ip.bin == '0b11111110100000000000000000000000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111'
    assert ip.bits() == '1111111010000000:0000000000000000:0000000000000000:0000000000000000:0000000000000000:0000000000000000:1101111010101101:1011111011101111'
    assert ip.words == (65152, 0, 0, 0, 0, 0, 57005, 48879)