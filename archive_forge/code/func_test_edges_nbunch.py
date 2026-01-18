import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_edges_nbunch(self):
    G = self.G()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
    pytest.raises(nx.NetworkXError, G.edges, 6)
    assert list(G.edges('Z')) == []
    assert list(G.edges([])) == []
    if G.is_directed():
        elist = [('A', 'B'), ('A', 'C'), ('B', 'D')]
    else:
        elist = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D')]
    assert edges_equal(list(G.edges(['A', 'B'])), elist)
    assert edges_equal(G.edges({'A', 'B'}), elist)
    G1 = self.G()
    G1.add_nodes_from('AB')
    assert edges_equal(G.edges(G1), elist)
    ndict = {'A': 'thing1', 'B': 'thing2'}
    assert edges_equal(G.edges(ndict), elist)
    assert edges_equal(list(G.edges('A')), [('A', 'B'), ('A', 'C')])
    assert nodes_equal(sorted(G), ['A', 'B', 'C', 'D'])
    assert edges_equal(list(G.edges()), [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])