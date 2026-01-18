import numpy as np
from pygsp import filters, utils
def compute_avg_adj_deg(G):
    """
    Compute the average adjacency degree for each node.

    The average adjacency degree is the average of the degrees of a node and
    its neighbors.

    Parameters
    ----------
    G: Graph
        Graph on which the statistic is extracted
    """
    return np.sum(np.dot(G.A, G.A), axis=1) / (np.sum(G.A, axis=1) + 1.0)