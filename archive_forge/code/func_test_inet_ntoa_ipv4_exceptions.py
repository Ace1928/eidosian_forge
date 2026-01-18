import pytest
from netaddr.fbsocket import inet_ntop, inet_pton, inet_ntoa, AF_INET, AF_INET6
def test_inet_ntoa_ipv4_exceptions():
    with pytest.raises(TypeError):
        inet_ntoa(1)
    with pytest.raises(ValueError):
        inet_ntoa('\x00')