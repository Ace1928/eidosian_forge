import numbers
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import type_desc
def bin_to_text(ip):
    """
    Converts binary representation to human readable IPv4 or IPv6 string.
    :param ip: binary representation of IPv4 or IPv6 address
    :return: IPv4 or IPv6 address string
    """
    if len(ip) == 4:
        return ipv4_to_str(ip)
    elif len(ip) == 16:
        return ipv6_to_str(ip)
    else:
        raise struct.error('Invalid ip address length: %s' % len(ip))