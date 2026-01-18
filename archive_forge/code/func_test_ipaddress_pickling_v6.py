import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipaddress_pickling_v6():
    ip = IPAddress('::ffff:192.0.2.1')
    assert ip == IPAddress('::ffff:192.0.2.1')
    assert ip.value == 281473902969345
    buf = pickle.dumps(ip)
    ip2 = pickle.loads(buf)
    assert ip2 == ip
    assert ip2.value == 281473902969345
    assert ip2.version == 6