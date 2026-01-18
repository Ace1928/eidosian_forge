import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
def min_weight(graph, src, dst, n):
    """Callback to handle merging nodes by choosing minimum weight.

    Returns a dictionary with `"weight"` set as either the weight between
    (`src`, `n`) or (`dst`, `n`) in `graph` or the minimum of the two when
    both exist.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The verices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dict with the `"weight"` attribute set the weight between
        (`src`, `n`) or (`dst`, `n`) in `graph` or the minimum of the two when
        both exist.

    """
    default = {'weight': np.inf}
    w1 = graph[n].get(src, default)['weight']
    w2 = graph[n].get(dst, default)['weight']
    return {'weight': min(w1, w2)}