from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def flipped(self, vertex):
    """
        Has this vertex been flipped?
        """
    return bool(len([e for e in self(vertex) if e[1] is vertex and e.slots[1] in (2, 3)]) % 2)