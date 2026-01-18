import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def check_vertex_cover(self, vertices):
    """Asserts that the given set of vertices is the vertex cover we
        expected from the bipartite graph constructed in the :meth:`setup`
        fixture.

        """
    assert len(vertices) == 5
    for u, v in self.graph.edges():
        assert u in vertices or v in vertices