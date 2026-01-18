from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def incoming_under(self, vertex):
    first, second, even_over = vertex
    incoming = [e.PD_index() for e in self(vertex) if e[1] is vertex]
    incoming.sort(key=lambda x: x % 2)
    return incoming[0] if even_over else incoming[1]