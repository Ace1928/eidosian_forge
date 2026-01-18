import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _raveled_offsets_and_distances
from ..util._map_array import map_array
def edge_function(x, y, distances):
    return distances