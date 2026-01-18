import pickle
import types
import random
import pytest
from netaddr import (
@pytest.mark.parametrize('address', ['0177.01', '0x7f.0.01', '10', '10.1', '10.0.1', '010.0.0.1', '10.01.0.1', '10.0.00.1', '10.0.1.01'])
def test_ipaddress_inet_pton_constructor_v4_rejects_invalid_input(address):
    with pytest.raises(AddrFormatError):
        IPAddress(address, flags=INET_PTON)
    with pytest.raises(AddrFormatError):
        IPAddress(address)