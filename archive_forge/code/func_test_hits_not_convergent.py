import pytest
import networkx as nx
from networkx.algorithms.link_analysis.hits_alg import (
def test_hits_not_convergent(self):
    G = nx.path_graph(50)
    with pytest.raises(nx.PowerIterationFailedConvergence):
        _hits_scipy(G, max_iter=1)
    with pytest.raises(nx.PowerIterationFailedConvergence):
        _hits_python(G, max_iter=1)
    with pytest.raises(nx.PowerIterationFailedConvergence):
        _hits_scipy(G, max_iter=0)
    with pytest.raises(nx.PowerIterationFailedConvergence):
        _hits_python(G, max_iter=0)
    with pytest.raises(ValueError):
        nx.hits(G, max_iter=0)
    with pytest.raises(sp.sparse.linalg.ArpackNoConvergence):
        nx.hits(G, max_iter=1)