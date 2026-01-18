import struct
from . import packet_base
from . import vlan
from . import mpls
from . import ether_types as ether
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
@classmethod
def get_packet_type(cls, type_):
    """Override method for the ethernet IEEE802.3 Length/Type
        field (self.ethertype).

        If the value of Length/Type field is less than or equal to
        1500 decimal(05DC hexadecimal), it means Length interpretation
        and be passed to the LLC sublayer."""
    if type_ <= ether.ETH_TYPE_IEEE802_3:
        type_ = ether.ETH_TYPE_IEEE802_3
    return cls._TYPES.get(type_)