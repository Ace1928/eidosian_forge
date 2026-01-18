from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def embed_arc(self):
    G = self.fat_graph
    v = self.get_incomplete_vertex()
    if v is None:
        return False
    if G.marked_valences[v] == 2:
        try:
            first, last, arc_edges = G.bridge(G.marked_arc(v))
        except ValueError:
            arc_edges, last = G.unmarked_arc(v)
            first = v
        self.do_flips(first, arc_edges[0], last, arc_edges[-1])
    else:
        arc_edges, last_vertex = G.unmarked_arc(v)
        self.do_flips(last_vertex, arc_edges[-1], v, arc_edges[0])
    G.mark(arc_edges)
    return True