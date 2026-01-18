import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
def map_to_link(map):
    num_edges = len(map) // 2
    crossings = [links.Crossing() for i in range(num_edges // 2)]
    for e in range(1, num_edges + 1):
        (a, i), (b, j) = (map[e], map[-e])
        crossings[a][i] = crossings[b][j]
    return links.Link(crossings, check_planarity=False)