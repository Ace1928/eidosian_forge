import pytest
from netaddr import valid_nmap_range, iter_nmap_range, IPAddress, AddrFormatError
def test_iter_nmap_range_invalid():
    with pytest.raises(AddrFormatError):
        list(iter_nmap_range('fe80::/64'))