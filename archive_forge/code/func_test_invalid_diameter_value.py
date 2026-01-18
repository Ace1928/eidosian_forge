import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_invalid_diameter_value(self):
    with pytest.raises(nx.NetworkXException, match='.*p must be >= 1'):
        nx.navigable_small_world_graph(5, p=0, q=0, dim=1)