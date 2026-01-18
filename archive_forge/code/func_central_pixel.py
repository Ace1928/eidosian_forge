import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _raveled_offsets_and_distances
from ..util._map_array import map_array
def central_pixel(graph, nodes=None, shape=None, partition_size=100):
    """Find the pixel with the highest closeness centrality.

    Closeness centrality is the inverse of the total sum of shortest distances
    from a node to every other node.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        The sparse matrix representation of the graph.
    nodes : array of int
        The raveled index of each node in graph in the image. If not provided,
        the returned value will be the index in the input graph.
    shape : tuple of int
        The shape of the image in which the nodes are embedded. If provided,
        the returned coordinates are a NumPy multi-index of the same
        dimensionality as the input shape. Otherwise, the returned coordinate
        is the raveled index provided in `nodes`.
    partition_size : int
        This function computes the shortest path distance between every pair
        of nodes in the graph. This can result in a very large (N*N) matrix.
        As a simple performance tweak, the distance values are computed in
        lots of `partition_size`, resulting in a memory requirement of only
        partition_size*N.

    Returns
    -------
    position : int or tuple of int
        If shape is given, the coordinate of the central pixel in the image.
        Otherwise, the raveled index of that pixel.
    distances : array of float
        The total sum of distances from each node to each other reachable
        node.
    """
    if nodes is None:
        nodes = np.arange(graph.shape[0])
    if partition_size is None:
        num_splits = 1
    else:
        num_splits = max(2, graph.shape[0] // partition_size)
    idxs = np.arange(graph.shape[0])
    total_shortest_path_len_list = []
    for partition in np.array_split(idxs, num_splits):
        shortest_paths = csgraph.shortest_path(graph, directed=False, indices=partition)
        shortest_paths_no_inf = np.nan_to_num(shortest_paths)
        total_shortest_path_len_list.append(np.sum(shortest_paths_no_inf, axis=1))
    total_shortest_path_len = np.concatenate(total_shortest_path_len_list)
    nonzero = np.flatnonzero(total_shortest_path_len)
    min_sp = np.argmin(total_shortest_path_len[nonzero])
    raveled_index = nodes[nonzero[min_sp]]
    if shape is not None:
        central = np.unravel_index(raveled_index, shape)
    else:
        central = raveled_index
    return (central, total_shortest_path_len)