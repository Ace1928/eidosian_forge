import pytest
from netaddr import valid_nmap_range, iter_nmap_range, IPAddress, AddrFormatError
def test_iter_nmap_range_with_multiple_targets_including_cidr():
    assert list(iter_nmap_range('192.168.0.0/29', '192.168.3-5,7.1', 'fe80::1')) == [IPAddress('192.168.0.0'), IPAddress('192.168.0.1'), IPAddress('192.168.0.2'), IPAddress('192.168.0.3'), IPAddress('192.168.0.4'), IPAddress('192.168.0.5'), IPAddress('192.168.0.6'), IPAddress('192.168.0.7'), IPAddress('192.168.3.1'), IPAddress('192.168.4.1'), IPAddress('192.168.5.1'), IPAddress('192.168.7.1'), IPAddress('fe80::1')]