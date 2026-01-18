import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_negative_beta_S1():
    with pytest.raises(nx.NetworkXError, match='The parameter beta cannot be smaller or equal to 0.'):
        G = nx.geometric_soft_configuration_graph(beta=-1, n=100, gamma=2.3, mean_degree=10, seed=42)