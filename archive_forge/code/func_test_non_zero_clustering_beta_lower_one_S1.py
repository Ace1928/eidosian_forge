import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_non_zero_clustering_beta_lower_one_S1():
    G = nx.geometric_soft_configuration_graph(beta=0.5, n=100, gamma=3.5, mean_degree=10, seed=42)
    assert nx.average_clustering(G) > 0