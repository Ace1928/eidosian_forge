from collections import defaultdict
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils.graph import single_source_shortest_path_length
def generate_graph(N=20):
    rng = np.random.RandomState(0)
    dist_matrix = rng.random_sample((N, N))
    dist_matrix = dist_matrix + dist_matrix.T
    i = (rng.randint(N, size=N * N // 2), rng.randint(N, size=N * N // 2))
    dist_matrix[i] = 0
    dist_matrix.flat[::N + 1] = 0
    return dist_matrix