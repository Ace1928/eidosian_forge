import pytest
import networkx as nx
def test_arf_layout_partial_input_test(self):
    """
        Checks whether partial pos input still returns a proper position.
        """
    G = self.Gs
    node = nx.utils.arbitrary_element(G)
    pos = nx.circular_layout(G)
    del pos[node]
    pos = nx.arf_layout(G, pos=pos)
    assert len(pos) == len(G)