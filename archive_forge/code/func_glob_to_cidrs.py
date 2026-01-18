from netaddr.core import AddrFormatError, AddrConversionError
from netaddr.ip import IPRange, IPAddress, IPNetwork, iprange_to_cidrs
def glob_to_cidrs(ipglob):
    """
    A function that accepts a glob-style IP range and returns a list of one
    or more IP CIDRs that exactly matches it.

    :param ipglob: an IP address range in a glob-style format.

    :return: a list of one or more IP objects.
    """
    return iprange_to_cidrs(*glob_to_iptuple(ipglob))