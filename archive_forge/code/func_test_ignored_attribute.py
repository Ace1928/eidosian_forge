import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_ignored_attribute(self):
    import io
    G = nx.Graph()
    fh = io.BytesIO()
    G.add_node(1, int_attr=1)
    G.add_node(2, empty_attr='  ')
    G.add_edge(1, 2, int_attr=2)
    G.add_edge(2, 3, empty_attr='  ')
    import warnings
    with warnings.catch_warnings(record=True) as w:
        nx.write_pajek(G, fh)
        assert len(w) == 4