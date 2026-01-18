import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_undirected(self):
    GG = self.XG.to_undirected()
    GG['u']['x']['weight'] = 2
    GG['y']['v']['weight'] = 2
    assert nx.astar_path(GG, 's', 'v') == ['s', 'x', 'u', 'v']
    assert nx.astar_path_length(GG, 's', 'v') == 8