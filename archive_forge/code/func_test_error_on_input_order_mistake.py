import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_error_on_input_order_mistake(self):
    pytest.raises(TypeError, self.tsp, self.UG, weight='weight')
    pytest.raises(nx.NetworkXError, self.tsp, self.UG, 'weight')