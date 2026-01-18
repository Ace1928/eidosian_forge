from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
def ipv6(self, prefix):
    """
        .. note:: This poses security risks in certain scenarios.             Please read RFC 4941 for details. Reference: RFCs 4291 and 4941.

        :param prefix: ipv6 prefix

        :return: new IPv6 `IPAddress` object based on this `EUI`             using the technique described in RFC 4291.
        """
    int_val = int(prefix) + int(self.modified_eui64())
    return IPAddress(int_val, version=6)