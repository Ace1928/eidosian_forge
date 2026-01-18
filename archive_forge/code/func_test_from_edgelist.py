import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist(self):
    G = nx.cycle_graph(10)
    G.add_weighted_edges_from(((u, v, u) for u, v in list(G.edges)))
    edgelist = nx.to_edgelist(G)
    source = [s for s, t, d in edgelist]
    target = [t for s, t, d in edgelist]
    weight = [d['weight'] for s, t, d in edgelist]
    edges = pd.DataFrame({'source': source, 'target': target, 'weight': weight})
    GG = nx.from_pandas_edgelist(edges, edge_attr='weight')
    assert nodes_equal(G.nodes(), GG.nodes())
    assert edges_equal(G.edges(), GG.edges())
    GW = nx.to_networkx_graph(edges, create_using=nx.Graph)
    assert nodes_equal(G.nodes(), GW.nodes())
    assert edges_equal(G.edges(), GW.edges())