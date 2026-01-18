import pytest
from netaddr import cidr_abbrev_to_verbose
from netaddr.strategy.ipv4 import expand_partial_address
def test_cidr_abbrev_to_verbose_invalid_prefixlen():
    assert cidr_abbrev_to_verbose('192.0.2.0/33') == '192.0.2.0/33'