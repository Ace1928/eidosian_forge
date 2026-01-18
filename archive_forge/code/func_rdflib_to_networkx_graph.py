from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List
def rdflib_to_networkx_graph(graph: Graph, calc_weights: bool=True, edge_attrs=lambda s, p, o: {'triples': [(s, p, o)]}, **kwds):
    """Converts the given graph into a networkx.Graph.

    As an rdflib.Graph() can contain multiple directed edges between nodes, by
    default adds the a 'triples' attribute to the single DiGraph edge with a
    list of triples between s and o in graph.
    Also by default calculates the edge weight as the len(triples).

    :Parameters:

        - graph: a rdflib.Graph.
        - calc_weights: If true calculate multi-graph edge-count as edge 'weight'
        - edge_attrs: Callable to construct later edge_attributes. It receives
                    3 variables (s, p, o) and should construct a dictionary that is
                    passed to networkx's add_edge(s, o, \\*\\*attrs) function.

                    By default this will include setting the 'triples' attribute here,
                    which is treated specially by us to be merged. Other attributes of
                    multi-edges will only contain the attributes of the first edge.
                    If you don't want the 'triples' attribute for tracking, set this to
                    ``lambda s, p, o: {}``.

    Returns:
        networkx.Graph

    >>> from rdflib import Graph, URIRef, Literal
    >>> g = Graph()
    >>> a, b, l = URIRef('a'), URIRef('b'), Literal('l')
    >>> p, q = URIRef('p'), URIRef('q')
    >>> edges = [(a, p, b), (a, q, b), (b, p, a), (b, p, l)]
    >>> for t in edges:
    ...     g.add(t)
    ...
    >>> ug = rdflib_to_networkx_graph(g)
    >>> ug[a][b]['weight']
    3
    >>> sorted(ug[a][b]['triples']) == [(a, p, b), (a, q, b), (b, p, a)]
    True
    >>> len(ug.edges())
    2
    >>> ug.size()
    2
    >>> ug.size(weight='weight')
    4.0

    >>> ug = rdflib_to_networkx_graph(g, False, edge_attrs=lambda s,p,o:{})
    >>> 'weight' in ug[a][b]
    False
    >>> 'triples' in ug[a][b]
    False
    """
    import networkx as nx
    g = nx.Graph()
    _rdflib_to_networkx_graph(graph, g, calc_weights, edge_attrs, **kwds)
    return g