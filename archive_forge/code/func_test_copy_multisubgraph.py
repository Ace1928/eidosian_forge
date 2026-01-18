import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_copy_multisubgraph(self):
    G = self.MG.copy()
    SG = G.subgraph([4, 5, 6])
    CSG = SG.copy(as_view=True)
    DCSG = SG.copy(as_view=False)
    assert hasattr(CSG, '_graph')
    assert not hasattr(DCSG, '_graph')