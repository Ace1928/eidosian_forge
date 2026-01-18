import pytest
from netaddr.ip import IPNetwork
from netaddr.contrib.subnet_splitter import SubnetSplitter
def test_ip_splitter_remove_prefix_larger_than_input_range():
    s = SubnetSplitter('172.24.0.0/16')
    assert s.available_subnets() == [IPNetwork('172.24.0.0/16')]
    assert s.extract_subnet(15, count=1) == []
    assert s.available_subnets() == [IPNetwork('172.24.0.0/16')]