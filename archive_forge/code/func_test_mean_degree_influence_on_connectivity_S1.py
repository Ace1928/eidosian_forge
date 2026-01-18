import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_mean_degree_influence_on_connectivity_S1():
    low_mean_degree = 2
    high_mean_degree = 20
    G_low = nx.geometric_soft_configuration_graph(beta=1.2, n=100, gamma=2.7, mean_degree=low_mean_degree, seed=42)
    G_high = nx.geometric_soft_configuration_graph(beta=1.2, n=100, gamma=2.7, mean_degree=high_mean_degree, seed=42)
    assert nx.number_connected_components(G_low) > nx.number_connected_components(G_high)