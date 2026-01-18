import struct
from . import packet_base
from os_ken.lib import type_desc
def label_from_bin(buf):
    """
    Converts binary representation label to integer.

    :param buf: Binary representation of label.
    :return: MPLS Label and BoS bit.
    """
    mpls_label = type_desc.Int3.to_user(bytes(buf))
    return (mpls_label >> 4, mpls_label & 1)