import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_invalid_exponent_for_decaying_probability_value(self):
    with pytest.raises(nx.NetworkXException, match='.*r must be >= 0'):
        nx.navigable_small_world_graph(5, p=1, q=0, r=-1, dim=1)