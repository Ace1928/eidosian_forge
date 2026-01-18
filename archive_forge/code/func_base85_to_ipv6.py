from netaddr.core import AddrFormatError
from netaddr.ip import IPAddress
def base85_to_ipv6(addr):
    """
    Convert a base 85 IPv6 address to its hexadecimal format.
    """
    tokens = list(addr)
    if len(tokens) != 20:
        raise AddrFormatError('Invalid base 85 IPv6 address: %r' % (addr,))
    result = 0
    for i, num in enumerate(reversed(tokens)):
        num = BASE_85_DICT[num]
        result += num * 85 ** i
    ip = IPAddress(result, 6)
    return str(ip)