from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
@property
def iab(self):
    """
        If is_iab() is True, the IAB (Individual Address Block) is returned,
        ``None`` otherwise.
        """
    if self.is_iab():
        return IAB(self._value >> 12)