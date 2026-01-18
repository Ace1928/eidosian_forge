from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def hex_signed_DT(self):
    """
        Return the hex encoding of the signed DT byte sequence.

        >>> d = DTcodec([(-6,-8,-2,-4)])
        >>> d2 = DTcodec(d.hex_signed_DT())
        >>> d2.code
        [(-6, -8, -2, -4)]
        """
    return '0x' + ''.join(('%.2x' % b for b in bytearray(self.signed_DT())))