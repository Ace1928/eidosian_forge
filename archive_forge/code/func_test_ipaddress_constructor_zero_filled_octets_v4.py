import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_constructor_zero_filled_octets_v4():
    assert IPAddress('010.000.000.001', flags=INET_ATON) == IPAddress('8.0.0.1')
    assert IPAddress('010.000.000.001', flags=INET_ATON | ZEROFILL) == IPAddress('10.0.0.1')
    assert IPAddress('010.000.001', flags=INET_ATON | ZEROFILL) == IPAddress('10.0.0.1')
    with pytest.raises(AddrFormatError):
        assert IPAddress('010.000.001', flags=ZEROFILL)
    with pytest.raises(AddrFormatError):
        assert IPAddress('010.000.001', flags=INET_PTON | ZEROFILL)
    assert IPAddress('010.000.000.001', flags=INET_PTON | ZEROFILL) == IPAddress('10.0.0.1')