import networkx as nx
from collections import deque
def strong_components(self):
    """
        Return the vertex sets of the strongly connected components.

        >>> G = Digraph([(0,1),(0,2),(1,2),(2,3),(3,1)])
        >>> G.strong_components() == [frozenset([1, 2, 3]), frozenset([0])]
        True
        >>> G = Digraph([(0,1),(0,2),(1,2),(2,3),(1,3)])
        >>> G.strong_components() == [frozenset([3]), frozenset([2]), frozenset([1]), frozenset([0])]
        True
        """
    return StrongConnector(self).components