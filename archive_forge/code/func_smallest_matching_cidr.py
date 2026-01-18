import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def smallest_matching_cidr(ip, cidrs):
    """
    Matches an IP address or subnet against a given sequence of IP addresses
    and subnets.

    :param ip: a single IP address or subnet.

    :param cidrs: a sequence of IP addresses and/or subnets.

    :return: the smallest (most specific) matching IPAddress or IPNetwork
        object from the provided sequence, None if there was no match.
    """
    match = None
    if not hasattr(cidrs, '__iter__'):
        raise TypeError('IP address/subnet sequence expected, not %r!' % (cidrs,))
    ip = IPAddress(ip)
    for cidr in sorted([IPNetwork(cidr) for cidr in cidrs]):
        if ip in cidr:
            match = cidr
        elif match is not None and cidr.network not in match:
            break
    return match