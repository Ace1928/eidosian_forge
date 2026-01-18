import struct
from . import packet_base
from os_ken.lib import type_desc
def label_to_bin(mpls_label, is_bos=True):
    """
    Converts integer label to binary representation.

    :param mpls_label: MPLS Label.
    :param is_bos: BoS bit.
    :return: Binary representation of label.
    """
    return type_desc.Int3.from_user(mpls_label << 4 | is_bos)