import networkx as nx
import numpy as np
from scipy.sparse import linalg
from . import _ncut, _ncut_cy
def cut_normalized(labels, rag, thresh=0.001, num_cuts=10, in_place=True, max_edge=1.0, *, rng=None):
    """Perform Normalized Graph cut on the Region Adjacency Graph.

    Given an image's labels and its similarity RAG, recursively perform
    a 2-way normalized cut on it. All nodes belonging to a subgraph
    that cannot be cut further are assigned a unique label in the
    output.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    num_cuts : int
        The number or N-cuts to perform before determining the optimal one.
    in_place : bool
        If set, modifies `rag` in place. For each node `n` the function will
        set a new attribute ``rag.nodes[n]['ncut label']``.
    max_edge : float, optional
        The maximum possible value of an edge in the RAG. This corresponds to
        an edge between identical regions. This is used to put self
        edges in the RAG.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

        The `rng` is used to determine the starting point
        of `scipy.sparse.linalg.eigsh`.

    Returns
    -------
    out : ndarray
        The new labeled array.

    Examples
    --------
    >>> from skimage import data, segmentation, graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels, mode='similarity')
    >>> new_labels = graph.cut_normalized(labels, rag)

    References
    ----------
    .. [1] Shi, J.; Malik, J., "Normalized cuts and image segmentation",
           Pattern Analysis and Machine Intelligence,
           IEEE Transactions on, vol. 22, no. 8, pp. 888-905, August 2000.

    """
    rng = np.random.default_rng(rng)
    if not in_place:
        rag = rag.copy()
    for node in rag.nodes():
        rag.add_edge(node, node, weight=max_edge)
    _ncut_relabel(rag, thresh, num_cuts, rng)
    map_array = np.zeros(labels.max() + 1, dtype=labels.dtype)
    for n, d in rag.nodes(data=True):
        map_array[d['labels']] = d['ncut label']
    return map_array[labels]