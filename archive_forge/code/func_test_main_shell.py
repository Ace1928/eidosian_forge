import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_main_shell(self):
    main_shell_subgraph = nx.k_shell(self.H)
    assert sorted(main_shell_subgraph.nodes()) == [2, 4, 5, 6]