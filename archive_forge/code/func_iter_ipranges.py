import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
def iter_ipranges(self):
    """Generate the merged IPRanges for this IPSet.

        In contrast to self.iprange(), this will work even when the IPSet is
        not contiguous. Adjacent IPRanges will be merged together, so you
        get the minimal number of IPRanges.
        """
    sorted_ranges = [(cidr._module.version, cidr.first, cidr.last) for cidr in self.iter_cidrs()]
    for start, stop in _iter_merged_ranges(sorted_ranges):
        yield IPRange(start, stop)