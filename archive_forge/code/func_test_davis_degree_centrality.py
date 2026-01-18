import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_davis_degree_centrality(self):
    G = self.davis
    deg = bipartite.degree_centrality(G, self.top_nodes)
    answer = {'E8': 0.78, 'E9': 0.67, 'E7': 0.56, 'Nora Fayette': 0.57, 'Evelyn Jefferson': 0.57, 'Theresa Anderson': 0.57, 'E6': 0.44, 'Sylvia Avondale': 0.5, 'Laura Mandeville': 0.5, 'Brenda Rogers': 0.5, 'Katherina Rogers': 0.43, 'E5': 0.44, 'Helen Lloyd': 0.36, 'E3': 0.33, 'Ruth DeSand': 0.29, 'Verne Sanderson': 0.29, 'E12': 0.33, 'Myra Liddel': 0.29, 'E11': 0.22, 'Eleanor Nye': 0.29, 'Frances Anderson': 0.29, 'Pearl Oglethorpe': 0.21, 'E4': 0.22, 'Charlotte McDowd': 0.29, 'E10': 0.28, 'Olivia Carleton': 0.14, 'Flora Price': 0.14, 'E2': 0.17, 'E1': 0.17, 'Dorothy Murchison': 0.14, 'E13': 0.17, 'E14': 0.17}
    for node, value in answer.items():
        assert value == pytest.approx(deg[node], abs=0.01)