import pytest
import networkx as nx
def test_directed_disallowed(self):
    """Tests that attempting to create a configuration model graph
        using a directed graph yields an exception.

        """
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.configuration_model([], create_using=nx.DiGraph())