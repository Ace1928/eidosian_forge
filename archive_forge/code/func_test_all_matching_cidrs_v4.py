import random
from netaddr import (
def test_all_matching_cidrs_v4():
    assert all_matching_cidrs('192.0.2.32', ['0.0.0.0/0', '10.0.0.0/8', '192.0.0.0/8', '192.0.1.0/24', '192.0.2.0/24', '192.0.3.0/24']) == [IPNetwork('0.0.0.0/0'), IPNetwork('192.0.0.0/8'), IPNetwork('192.0.2.0/24')]