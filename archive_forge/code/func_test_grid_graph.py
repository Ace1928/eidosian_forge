import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_grid_graph():
    embedding_data = {(0, 1): [(0, 0), (1, 1), (0, 2)], (1, 2): [(1, 1), (2, 2), (0, 2)], (0, 0): [(0, 1), (1, 0)], (2, 1): [(2, 0), (2, 2), (1, 1)], (1, 1): [(2, 1), (1, 2), (0, 1), (1, 0)], (2, 0): [(1, 0), (2, 1)], (2, 2): [(1, 2), (2, 1)], (1, 0): [(0, 0), (2, 0), (1, 1)], (0, 2): [(1, 2), (0, 1)]}
    check_embedding_data(embedding_data)