import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist_multidigraph_and_edge_attr(self):
    edges = [('X1', 'X4', {'Co': 'zA', 'Mi': 0, 'St': 'X1'}), ('X1', 'X4', {'Co': 'zB', 'Mi': 54, 'St': 'X2'}), ('X1', 'X4', {'Co': 'zB', 'Mi': 49, 'St': 'X3'}), ('X1', 'X4', {'Co': 'zB', 'Mi': 44, 'St': 'X4'}), ('Y1', 'Y3', {'Co': 'zC', 'Mi': 0, 'St': 'Y1'}), ('Y1', 'Y3', {'Co': 'zC', 'Mi': 34, 'St': 'Y2'}), ('Y1', 'Y3', {'Co': 'zC', 'Mi': 29, 'St': 'X2'}), ('Y1', 'Y3', {'Co': 'zC', 'Mi': 24, 'St': 'Y3'}), ('Z1', 'Z3', {'Co': 'zD', 'Mi': 0, 'St': 'Z1'}), ('Z1', 'Z3', {'Co': 'zD', 'Mi': 14, 'St': 'X3'})]
    Gtrue = nx.MultiDiGraph(edges)
    data = {'O': ['X1', 'X1', 'X1', 'X1', 'Y1', 'Y1', 'Y1', 'Y1', 'Z1', 'Z1'], 'D': ['X4', 'X4', 'X4', 'X4', 'Y3', 'Y3', 'Y3', 'Y3', 'Z3', 'Z3'], 'St': ['X1', 'X2', 'X3', 'X4', 'Y1', 'Y2', 'X2', 'Y3', 'Z1', 'X3'], 'Co': ['zA', 'zB', 'zB', 'zB', 'zC', 'zC', 'zC', 'zC', 'zD', 'zD'], 'Mi': [0, 54, 49, 44, 0, 34, 29, 24, 0, 14]}
    df = pd.DataFrame.from_dict(data)
    G1 = nx.from_pandas_edgelist(df, source='O', target='D', edge_attr=True, create_using=nx.MultiDiGraph)
    G2 = nx.from_pandas_edgelist(df, source='O', target='D', edge_attr=['St', 'Co', 'Mi'], create_using=nx.MultiDiGraph)
    assert graphs_equal(G1, Gtrue)
    assert graphs_equal(G2, Gtrue)