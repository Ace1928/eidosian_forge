from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List
def rdflib_to_networkx_digraph(graph: Graph, calc_weights: bool=True, edge_attrs=lambda s, p, o: {'triples': [(s, p, o)]}, **kwds):
    """Converts the given graph into a networkx.DiGraph.

    As an rdflib.Graph() can contain multiple edges between nodes, by default
    adds the a 'triples' attribute to the single DiGraph edge with a list of
    all triples between s and o.
    Also by default calculates the edge weight as the length of triples.

    :Parameters:

        - ``graph``: a rdflib.Graph.
        - ``calc_weights``: If true calculate multi-graph edge-count as edge 'weight'
        - ``edge_attrs``: Callable to construct later edge_attributes. It receives
            3 variables (s, p, o) and should construct a dictionary that is passed to
            networkx's add_edge(s, o, \\*\\*attrs) function.

            By default this will include setting the 'triples' attribute here,
            which is treated specially by us to be merged. Other attributes of
            multi-edges will only contain the attributes of the first edge.
            If you don't want the 'triples' attribute for tracking, set this to
            ``lambda s, p, o: {}``.

    Returns: networkx.DiGraph

    >>> from rdflib import Graph, URIRef, Literal
    >>> g = Graph()
    >>> a, b, l = URIRef('a'), URIRef('b'), Literal('l')
    >>> p, q = URIRef('p'), URIRef('q')
    >>> edges = [(a, p, b), (a, q, b), (b, p, a), (b, p, l)]
    >>> for t in edges:
    ...     g.add(t)
    ...
    >>> dg = rdflib_to_networkx_digraph(g)
    >>> dg[a][b]['weight']
    2
    >>> sorted(dg[a][b]['triples']) == [(a, p, b), (a, q, b)]
    True
    >>> len(dg.edges())
    3
    >>> dg.size()
    3
    >>> dg.size(weight='weight')
    4.0

    >>> dg = rdflib_to_networkx_graph(g, False, edge_attrs=lambda s,p,o:{})
    >>> 'weight' in dg[a][b]
    False
    >>> 'triples' in dg[a][b]
    False

    """
    import networkx as nx
    dg = nx.DiGraph()
    _rdflib_to_networkx_graph(graph, dg, calc_weights, edge_attrs, **kwds)
    return dg