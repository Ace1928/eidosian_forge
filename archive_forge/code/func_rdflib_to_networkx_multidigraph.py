from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List
def rdflib_to_networkx_multidigraph(graph: Graph, edge_attrs=lambda s, p, o: {'key': p}, **kwds):
    """Converts the given graph into a networkx.MultiDiGraph.

    The subjects and objects are the later nodes of the MultiDiGraph.
    The predicates are used as edge keys (to identify multi-edges).

    :Parameters:

        - graph: a rdflib.Graph.
        - edge_attrs: Callable to construct later edge_attributes. It receives
            3 variables (s, p, o) and should construct a dictionary that is
            passed to networkx's add_edge(s, o, \\*\\*attrs) function.

            By default this will include setting the MultiDiGraph key=p here.
            If you don't want to be able to re-identify the edge later on, you
            can set this to ``lambda s, p, o: {}``. In this case MultiDiGraph's
            default (increasing ints) will be used.

    Returns:
        networkx.MultiDiGraph

    >>> from rdflib import Graph, URIRef, Literal
    >>> g = Graph()
    >>> a, b, l = URIRef('a'), URIRef('b'), Literal('l')
    >>> p, q = URIRef('p'), URIRef('q')
    >>> edges = [(a, p, b), (a, q, b), (b, p, a), (b, p, l)]
    >>> for t in edges:
    ...     g.add(t)
    ...
    >>> mdg = rdflib_to_networkx_multidigraph(g)
    >>> len(mdg.edges())
    4
    >>> mdg.has_edge(a, b)
    True
    >>> mdg.has_edge(a, b, key=p)
    True
    >>> mdg.has_edge(a, b, key=q)
    True

    >>> mdg = rdflib_to_networkx_multidigraph(g, edge_attrs=lambda s,p,o: {})
    >>> mdg.has_edge(a, b, key=0)
    True
    >>> mdg.has_edge(a, b, key=1)
    True
    """
    import networkx as nx
    mdg = nx.MultiDiGraph()
    _rdflib_to_networkx_graph(graph, mdg, False, edge_attrs, **kwds)
    return mdg