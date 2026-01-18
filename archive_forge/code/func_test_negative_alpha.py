import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
def test_negative_alpha(self):
    with pytest.raises(ValueError, match='alpha must be positive'):
        random_k_out_graph(10, 3, -1)