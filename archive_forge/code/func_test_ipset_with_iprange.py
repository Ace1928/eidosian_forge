import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_with_iprange():
    s1 = IPSet(['10.0.0.0/25', '10.0.0.128/25'])
    assert s1.iprange() == IPRange('10.0.0.0', '10.0.0.255')
    assert s1.iscontiguous()
    s1.remove('10.0.0.16')
    assert s1 == IPSet(['10.0.0.0/28', '10.0.0.17/32', '10.0.0.18/31', '10.0.0.20/30', '10.0.0.24/29', '10.0.0.32/27', '10.0.0.64/26', '10.0.0.128/25'])
    assert not s1.iscontiguous()
    with pytest.raises(ValueError):
        s1.iprange()
    assert list(s1.iter_ipranges()) == [IPRange('10.0.0.0', '10.0.0.15'), IPRange('10.0.0.17', '10.0.0.255')]
    s2 = IPSet(['0.0.0.0/0'])
    assert s2.iscontiguous()
    assert s2.iprange() == IPRange('0.0.0.0', '255.255.255.255')
    s3 = IPSet()
    assert s3.iscontiguous()
    assert s3.iprange() is None
    s4 = IPSet(IPRange('10.0.0.0', '10.0.0.8'))
    assert s4.iscontiguous()