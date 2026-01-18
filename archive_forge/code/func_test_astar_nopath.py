import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_nopath(self):
    with pytest.raises(nx.NodeNotFound):
        nx.astar_path(self.XG, 's', 'moon')