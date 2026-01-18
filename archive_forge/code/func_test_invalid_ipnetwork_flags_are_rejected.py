import weakref
import pytest
from netaddr import INET_ATON, INET_PTON, IPAddress, IPNetwork, IPRange, NOHOST
def test_invalid_ipnetwork_flags_are_rejected():
    with pytest.raises(ValueError):
        IPNetwork('1.2.0.0/16', flags=INET_PTON)