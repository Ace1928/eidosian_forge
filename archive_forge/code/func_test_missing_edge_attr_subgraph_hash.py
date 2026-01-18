import pytest
import networkx as nx
from networkx.generators import directed
def test_missing_edge_attr_subgraph_hash():
    """
    If the 'edge_attr' argument is supplied but is missing from an edge in the graph,
    we should raise a KeyError
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2, {'edge_attr1': 'a'}), (1, 3, {})])
    pytest.raises(KeyError, nx.weisfeiler_lehman_subgraph_hashes, G, edge_attr='edge_attr1')