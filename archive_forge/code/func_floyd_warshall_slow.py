from collections import defaultdict
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils.graph import single_source_shortest_path_length
def floyd_warshall_slow(graph, directed=False):
    N = graph.shape[0]
    graph[np.where(graph == 0)] = np.inf
    graph.flat[::N + 1] = 0
    if not directed:
        graph = np.minimum(graph, graph.T)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                graph[i, j] = min(graph[i, j], graph[i, k] + graph[k, j])
    graph[np.where(np.isinf(graph))] = 0
    return graph