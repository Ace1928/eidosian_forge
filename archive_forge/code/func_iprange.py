import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
def iprange(self):
    """
        Generates an IPRange for this IPSet, if all its members
        form a single contiguous sequence.

        Raises ``ValueError`` if the set is not contiguous.

        :return: An ``IPRange`` for all IPs in the IPSet.
        """
    if self.iscontiguous():
        cidrs = self.iter_cidrs()
        if not cidrs:
            return None
        return IPRange(cidrs[0][0], cidrs[-1][-1])
    else:
        raise ValueError('IPSet is not contiguous')