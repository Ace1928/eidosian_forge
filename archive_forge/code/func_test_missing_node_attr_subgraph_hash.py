import pytest
import networkx as nx
from networkx.generators import directed
def test_missing_node_attr_subgraph_hash():
    """
    If the 'node_attr' argument is supplied but is missing from a node in the graph,
    we should raise a KeyError
    """
    G = nx.Graph()
    G.add_nodes_from([(1, {'node_attr1': 'a'}), (2, {})])
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4)])
    pytest.raises(KeyError, nx.weisfeiler_lehman_subgraph_hashes, G, node_attr='node_attr1')