import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def all_matching_cidrs(ip, cidrs):
    """
    Matches an IP address or subnet against a given sequence of IP addresses
    and subnets.

    :param ip: a single IP address.

    :param cidrs: a sequence of IP addresses and/or subnets.

    :return: all matching IPAddress and/or IPNetwork objects from the provided
        sequence, an empty list if there was no match.
    """
    matches = []
    if not hasattr(cidrs, '__iter__'):
        raise TypeError('IP address/subnet sequence expected, not %r!' % (cidrs,))
    ip = IPAddress(ip)
    for cidr in sorted([IPNetwork(cidr) for cidr in cidrs]):
        if ip in cidr:
            matches.append(cidr)
        elif matches and cidr.network not in matches[-1]:
            break
    return matches