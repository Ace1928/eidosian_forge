import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
@classmethod
def construct(EdgeComponentAuxGraph, G):
    """Builds an auxiliary graph encoding edge-connectivity between nodes.

        Notes
        -----
        Given G=(V, E), initialize an empty auxiliary graph A.
        Choose an arbitrary source node s.  Initialize a set N of available
        nodes (that can be used as the sink). The algorithm picks an
        arbitrary node t from N - {s}, and then computes the minimum st-cut
        (S, T) with value w. If G is directed the minimum of the st-cut or
        the ts-cut is used instead. Then, the edge (s, t) is added to the
        auxiliary graph with weight w. The algorithm is called recursively
        first using S as the available nodes and s as the source, and then
        using T and t. Recursion stops when the source is the only available
        node.

        Parameters
        ----------
        G : NetworkX graph
        """
    not_implemented_for('multigraph')(lambda G: G)(G)

    def _recursive_build(H, A, source, avail):
        if {source} == avail:
            return
        sink = arbitrary_element(avail - {source})
        value, (S, T) = nx.minimum_cut(H, source, sink)
        if H.is_directed():
            value_, (T_, S_) = nx.minimum_cut(H, sink, source)
            if value_ < value:
                value, S, T = (value_, S_, T_)
        A.add_edge(source, sink, weight=value)
        _recursive_build(H, A, source, avail.intersection(S))
        _recursive_build(H, A, sink, avail.intersection(T))
    H = G.__class__()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges(), capacity=1)
    A = nx.Graph()
    if H.number_of_nodes() > 0:
        source = arbitrary_element(H.nodes())
        avail = set(H.nodes())
        _recursive_build(H, A, source, avail)
    self = EdgeComponentAuxGraph()
    self.A = A
    self.H = H
    return self