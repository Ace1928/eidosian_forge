import random
from netaddr import (
def test_largest_matching_cidr_v4():
    assert largest_matching_cidr('192.0.2.0', ['192.0.2.0']) == IPNetwork('192.0.2.0/32')
    assert largest_matching_cidr('192.0.2.0', ['10.0.0.1', '192.0.2.0']) == IPNetwork('192.0.2.0/32')
    assert largest_matching_cidr('192.0.2.0', ['10.0.0.1', '192.0.2.0', '224.0.0.1']) == IPNetwork('192.0.2.0/32')
    assert largest_matching_cidr('192.0.2.0', ['10.0.0.1', '224.0.0.1']) is None