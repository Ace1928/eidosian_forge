import random
from netaddr import (
def test_extended_cidr_merge():
    orig_cidr_ipv4 = IPNetwork('192.0.2.0/23')
    orig_cidr_ipv6 = IPNetwork('::192.0.2.0/120')
    cidr_subnets = [str(c) for c in orig_cidr_ipv4.subnet(28)] + list(orig_cidr_ipv4.subnet(28)) + [str(c) for c in orig_cidr_ipv6.subnet(124)] + list(orig_cidr_ipv6.subnet(124)) + ['192.0.2.1/32', '192.0.2.128/25', '::192.0.2.92/128']
    random.shuffle(cidr_subnets)
    merged_cidrs = cidr_merge(cidr_subnets)
    assert merged_cidrs == [IPNetwork('192.0.2.0/23'), IPNetwork('::192.0.2.0/120')]
    assert merged_cidrs == [orig_cidr_ipv4, orig_cidr_ipv6]