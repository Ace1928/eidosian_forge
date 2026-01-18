import pytest
import networkx as nx
def test_arf_layout_negative_a_check(self):
    """
        Checks input parameters correctly raises errors. For example,  `a` should be larger than 1
        """
    G = self.Gs
    pytest.raises(ValueError, nx.arf_layout, G=G, a=-1)