import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_k_core_multigraph(self):
    core_number = nx.core_number(self.H)
    H = nx.MultiGraph(self.H)
    with pytest.deprecated_call():
        nx.k_core(H, k=0, core_number=core_number)