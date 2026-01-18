import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
def iscontiguous(self):
    """
        Returns True if the members of the set form a contiguous IP
        address range (with no gaps), False otherwise.

        :return: ``True`` if the ``IPSet`` object is contiguous.
        """
    cidrs = self.iter_cidrs()
    if len(cidrs) > 1:
        previous = cidrs[0][0]
        for cidr in cidrs:
            if cidr[0] != previous:
                return False
            previous = cidr[-1] + 1
    return True