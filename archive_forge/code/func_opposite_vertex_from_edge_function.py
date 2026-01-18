from collections import OrderedDict
from ... import sage_helper
def opposite_vertex_from_edge_function(vertices):
    other = [v for v in range(3) if v not in vertices]
    assert len(vertices) == 2 and len(other) == 1
    return other[0]