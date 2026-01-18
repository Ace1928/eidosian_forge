from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
def modified_eui64(self):
    """
        - create a new EUI object with a modified EUI-64 as described in RFC 4291 section 2.5.1

        :return: a new and modified 64-bit EUI object.
        """
    eui64 = self.eui64()
    eui64._value ^= 144115188075855872
    return eui64