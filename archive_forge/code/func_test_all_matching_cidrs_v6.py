from netaddr import iprange_to_cidrs, IPNetwork, cidr_merge, all_matching_cidrs
def test_all_matching_cidrs_v6():
    assert all_matching_cidrs('::ffff:192.0.2.1', ['::ffff:192.0.2.0/96']) == [IPNetwork('::ffff:192.0.2.0/96')]
    assert all_matching_cidrs('::192.0.2.1', ['::192.0.2.0/96']) == [IPNetwork('::192.0.2.0/96')]
    assert all_matching_cidrs('::192.0.2.1', ['192.0.2.0/23']) == []
    assert all_matching_cidrs('::192.0.2.1', ['192.0.2.0/24', '::192.0.2.0/120']) == [IPNetwork('::192.0.2.0/120')]
    assert all_matching_cidrs('::192.0.2.1', [IPNetwork('192.0.2.0/24'), IPNetwork('::192.0.2.0/120')]) == [IPNetwork('::192.0.2.0/120')]