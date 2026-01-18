from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def marked_arc(self, vertex):
    """
        Given a vertex with marked valence 2, find the maximal marked
        arc containing the vertex for which all interior edges have
        marked valence 2.  If the marked subgraph is a circle, or a
        dead end is reached, raise :class:`ValueError`.  Return a list of
        edges in the arc.
        """
    left_path, right_path, vertices = ([], [], set())
    vertices.add(vertex)
    try:
        left, right = [e for e in self(vertex) if e.marked]
    except ValueError:
        raise RuntimeError('Vertex must have two marked edges.')
    for edge, path in ((left, left_path), (right, right_path)):
        V = vertex
        while True:
            path.append(edge)
            V = edge(V)
            if V == vertex:
                raise ValueError('Marked graph is a circle')
            edges = [e for e in self(V) if e.marked and e != edge]
            if len(edges) == 0:
                raise ValueError('Marked graph has a dead end at %s.' % V)
            if len(edges) > 1:
                break
            else:
                vertices.add(V)
                edge = edges.pop()
    left_path.reverse()
    return left_path + right_path