import weakref
import pytest
from netaddr import INET_ATON, INET_PTON, IPAddress, IPNetwork, IPRange, NOHOST
@pytest.mark.parametrize('flags', [NOHOST, INET_ATON | INET_PTON])
def test_invalid_ipaddress_flags_are_rejected(flags):
    with pytest.raises(ValueError):
        IPAddress('1.2.3.4', flags=flags)