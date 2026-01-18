import pytest
import networkx as nx
def test_is_regular_expander_badinput():
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    with pytest.raises(nx.NetworkXError, match='epsilon must be non negative'):
        nx.is_regular_expander(nx.Graph(), epsilon=-1)