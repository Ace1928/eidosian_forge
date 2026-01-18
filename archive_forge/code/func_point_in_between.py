import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def point_in_between(a, b, p):
    x1, y1 = a
    x2, y2 = b
    px, py = p
    dist_1_2 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    dist_1_p = math.sqrt((x1 - px) ** 2 + (y1 - py) ** 2)
    dist_2_p = math.sqrt((x2 - px) ** 2 + (y2 - py) ** 2)
    return is_close(dist_1_p + dist_2_p, dist_1_2)