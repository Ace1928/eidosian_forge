from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def find_circle(self, first_edge):
    """
        Follow a component, starting at the given (directed) edge,
        until the first time it crosses itself.  Throw away the tail
        to get a cycle.  Return the list of vertices and the list of
        edges traversed by the cycle.
        """
    edges = []
    vertices = [first_edge[0]]
    seen = set(vertices)
    for edge in self.fat_graph.path(first_edge[0], first_edge):
        vertex = edge[1]
        if vertex in seen:
            edges.append(edge)
            break
        else:
            seen.add(vertex)
            vertices.append(vertex)
            edges.append(edge)
    n = vertices.index(vertex)
    edges = edges[n:]
    vertices = vertices[n:]
    return (vertices, edges)