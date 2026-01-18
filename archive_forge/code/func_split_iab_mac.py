from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
@classmethod
def split_iab_mac(cls, eui_int, strict=False):
    """
        :param eui_int: a MAC IAB as an unsigned integer.

        :param strict: If True, raises a ValueError if the last 12 bits of
            IAB MAC/EUI-48 address are non-zero, ignores them otherwise.
            (Default: False)
        """
    if eui_int >> 12 in cls.IAB_EUI_VALUES:
        return (eui_int, 0)
    user_mask = 2 ** 12 - 1
    iab_mask = 2 ** 48 - 1 ^ user_mask
    iab_bits = eui_int >> 12
    user_bits = (eui_int | iab_mask) - iab_mask
    if iab_bits >> 12 in cls.IAB_EUI_VALUES:
        if strict and user_bits != 0:
            raise ValueError('%r is not a strict IAB!' % hex(user_bits))
    else:
        raise ValueError('%r is not an IAB address!' % hex(eui_int))
    return (iab_bits, user_bits)