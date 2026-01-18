import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
def random_map(num_verts, edge_conn_param=4, num_link_comps=0, max_tries=100):
    """
    Returns a dictionary of endpoints of edges in the form:

    (signed edge) -> (vertex, position)
    """
    if isinstance(num_verts, list):
        data = num_verts
    else:
        data = raw_random_map(num_verts, edge_conn_param, num_link_comps, max_tries)
        if data is None:
            raise LinkGenerationError(max_tries)
    vertex_adjacencies = []
    for vertex, adjacencies in data:
        if random.randrange(2):
            adjacencies = adjacencies[1:] + adjacencies[:1]
        vertex_adjacencies.append(adjacencies)
    edge_adjacencies = {}
    for v, edges in enumerate(vertex_adjacencies):
        for i, edge in enumerate(edges):
            edge_adjacencies[edge] = (v, i)
    return edge_adjacencies