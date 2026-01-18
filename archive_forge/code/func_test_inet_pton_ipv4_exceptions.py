import pytest
from netaddr.fbsocket import inet_ntop, inet_pton, inet_ntoa, AF_INET, AF_INET6
def test_inet_pton_ipv4_exceptions():
    with pytest.raises(OSError):
        inet_pton(AF_INET, '::0x07f')
    with pytest.raises(TypeError):
        inet_pton(AF_INET, 1)