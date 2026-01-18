import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_adjacency_named(self):
    data = {'A': {'A': 0, 'B': 0, 'C': 0}, 'B': {'A': 1, 'B': 0, 'C': 0}, 'C': {'A': 0, 'B': 1, 'C': 0}}
    dftrue = pd.DataFrame(data, dtype=np.intp)
    df = dftrue[['A', 'C', 'B']]
    G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
    df = nx.to_pandas_adjacency(G, dtype=np.intp)
    pd.testing.assert_frame_equal(df, dftrue)