import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_unions():
    assert IPSet(['192.0.2.0']) == IPSet(['192.0.2.0/32'])
    assert IPSet(['192.0.2.0']) | IPSet(['192.0.2.1']) == IPSet(['192.0.2.0/31'])
    assert IPSet(['192.0.2.0']) | IPSet(['192.0.2.1']) | IPSet(['192.0.2.3']) == IPSet(['192.0.2.0/31', '192.0.2.3/32'])
    assert IPSet(['192.0.2.0']) | IPSet(['192.0.2.1']) | IPSet(['192.0.2.3/30']) == IPSet(['192.0.2.0/30'])
    assert IPSet(['192.0.2.0']) | IPSet(['192.0.2.1']) | IPSet(['192.0.2.3/31']) == IPSet(['192.0.2.0/30'])
    assert IPSet(['192.0.2.0/24']) | IPSet(['192.0.3.0/24']) | IPSet(['192.0.4.0/24']) == IPSet(['192.0.2.0/23', '192.0.4.0/24'])