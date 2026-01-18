from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def level_transformations(self):
    """
        Generator for non-trivial level transformations.

        >>> P = Presentation(['AABCaBacAcbabC'])
        >>> relators = []
        >>> for x, X in P.level_transformations():
        ...   P = Presentation(['AABCaBacAcbabC'])
        ...   P = P.whitehead_move(x, X)
        ...   relators += P.relators
        >>> sorted(relators)
        [AAABCBaaccbabC, AABaBcacAbaCbC, AABCaBaaccbbAC, AACaBacBAcabbC, ABCaaBcAAcbabC, ABCaaBacAcbbAC]
        """
    reducers, levels = self.find_reducers()
    if reducers:
        raise ValueError('Presentation is not minimal.')
    for generator, cut in levels:
        edges = set()
        for weight, path in cut['paths']:
            for vertex, edge in path:
                edges.add((vertex, edge(vertex)))
        for edge in cut['unsaturated']:
            x, y = edge
            edges.add((x, y))
            edges.add((y, x))
        D = Digraph(edges)
        P = Poset(D.component_DAG())
        for subset in P.closed_subsets():
            if 1 < len(subset) < len(P) - 1:
                yield (generator, frozenset.union(*subset))