from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def entry_slot(self, N):
    if N == self[0]:
        return South
    elif N == self[1]:
        return East
    else:
        raise ValueError('%d is not a label of %s' % (N, self))