from . import base
from . import graphs
from scipy import sparse
import numpy as np
import pickle
import pygsp
import tasklogger
import warnings
def from_igraph(G, attribute='weight', **kwargs):
    """Convert an igraph.Graph to a graphtools.Graph

    Creates a graphtools.graphs.TraditionalGraph with a
    precomputed adjacency matrix

    Parameters
    ----------
    G : igraph.Graph
        Graph to be converted
    attribute : str, optional (default: "weight")
        attribute containing edge weights, if any.
        If None, unweighted graph is built
    kwargs
        keyword arguments for graphtools.Graph

    Returns
    -------
    G : graphtools.graphs.TraditionalGraph
    """
    if 'precomputed' in kwargs:
        if kwargs['precomputed'] != 'adjacency':
            warnings.warn("Cannot build graph from igraph with precomputed={}. Use 'adjacency' instead.".format(kwargs['precomputed']), UserWarning)
        del kwargs['precomputed']
    try:
        K = G.get_adjacency(attribute=attribute).data
    except ValueError as e:
        if str(e) == 'Attribute does not exist':
            warnings.warn('Edge attribute {} not found. Returning unweighted graph'.format(attribute), UserWarning)
        K = G.get_adjacency(attribute=None).data
    return Graph(sparse.coo_matrix(K), precomputed='adjacency', **kwargs)