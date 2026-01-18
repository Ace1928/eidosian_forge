import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def ipv4(self):
    """
        :return: A numerically equivalent version 4 `IPNetwork` object.             Raises an `AddrConversionError` if IPv6 address cannot be             converted to IPv4.
        """
    ip = None
    klass = self.__class__
    if self._module.version == 4:
        ip = klass('%s/%d' % (self.ip, self.prefixlen))
    elif self._module.version == 6:
        if 0 <= self._value <= _ipv4.max_int:
            addr = _ipv4.int_to_str(self._value)
            ip = klass('%s/%d' % (addr, self.prefixlen - 96))
        elif _ipv4.max_int <= self._value <= 281474976710655:
            addr = _ipv4.int_to_str(self._value - 281470681743360)
            ip = klass('%s/%d' % (addr, self.prefixlen - 96))
        else:
            raise AddrConversionError('IPv6 address %s unsuitable for conversion to IPv4!' % self)
    return ip