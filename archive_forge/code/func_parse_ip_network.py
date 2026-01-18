import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def parse_ip_network(module, addr, flags=0):
    if isinstance(addr, tuple):
        if len(addr) != 2:
            raise AddrFormatError('invalid %s tuple!' % module.family_name)
        value, prefixlen = addr
        if not 0 <= value <= module.max_int:
            raise AddrFormatError('invalid address value for %s tuple!' % module.family_name)
        if not 0 <= prefixlen <= module.width:
            raise AddrFormatError('invalid prefix for %s tuple!' % module.family_name)
    elif isinstance(addr, str):
        if '/' in addr:
            val1, val2 = addr.split('/', 1)
        else:
            val1 = addr
            val2 = None
        ip = IPAddress(val1, module.version, flags=INET_PTON)
        value = ip._value
        try:
            prefixlen = int(val2)
        except TypeError:
            if val2 is None:
                prefixlen = module.width
        except ValueError:
            mask = IPAddress(val2, module.version, flags=INET_PTON)
            if mask.is_netmask():
                prefixlen = module.netmask_to_prefix[mask._value]
            elif mask.is_hostmask():
                prefixlen = module.hostmask_to_prefix[mask._value]
            else:
                raise AddrFormatError('addr %r is not a valid IPNetwork!' % addr)
        if not 0 <= prefixlen <= module.width:
            raise AddrFormatError('invalid prefix for %s address!' % module.family_name)
    else:
        raise TypeError('unexpected type %s for addr arg' % type(addr))
    if flags & NOHOST:
        netmask = module.prefix_to_netmask[prefixlen]
        value = value & netmask
    return (value, prefixlen)