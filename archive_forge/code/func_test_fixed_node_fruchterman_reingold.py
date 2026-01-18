import pytest
import networkx as nx
def test_fixed_node_fruchterman_reingold(self):
    pos = nx.circular_layout(self.Gi)
    npos = nx.spring_layout(self.Gi, pos=pos, fixed=[(0, 0)])
    assert tuple(pos[0, 0]) == tuple(npos[0, 0])
    pos = nx.circular_layout(self.bigG)
    npos = nx.spring_layout(self.bigG, pos=pos, fixed=[(0, 0)])
    for axis in range(2):
        assert pos[0, 0][axis] == pytest.approx(npos[0, 0][axis], abs=1e-07)