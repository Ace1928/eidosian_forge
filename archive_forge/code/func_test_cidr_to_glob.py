from netaddr import (
def test_cidr_to_glob():
    assert cidr_to_glob('10.0.0.1/32') == '10.0.0.1'
    assert cidr_to_glob('192.0.2.0/24') == '192.0.2.*'
    assert cidr_to_glob('172.16.0.0/12') == '172.16-31.*.*'
    assert cidr_to_glob('0.0.0.0/0') == '*.*.*.*'