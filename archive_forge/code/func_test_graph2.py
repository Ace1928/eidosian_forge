import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_graph2():
    embedding_data = {0: [8, 6], 1: [2, 6, 9], 2: [8, 1, 7, 9, 6, 4], 3: [9], 4: [2], 5: [6, 8], 6: [9, 1, 0, 5, 2], 7: [9, 2], 8: [0, 2, 5], 9: [1, 6, 2, 7, 3]}
    check_embedding_data(embedding_data)