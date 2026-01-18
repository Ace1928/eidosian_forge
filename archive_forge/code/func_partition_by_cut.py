import networkx as nx
import numpy as np
from scipy.sparse import linalg
from . import _ncut, _ncut_cy
def partition_by_cut(cut, rag):
    """Compute resulting subgraphs from given bi-partition.

    Parameters
    ----------
    cut : array
        A array of booleans. Elements set to `True` belong to one
        set.
    rag : RAG
        The Region Adjacency Graph.

    Returns
    -------
    sub1, sub2 : RAG
        The two resulting subgraphs from the bi-partition.
    """
    nodes1 = [n for i, n in enumerate(rag.nodes()) if cut[i]]
    nodes2 = [n for i, n in enumerate(rag.nodes()) if not cut[i]]
    sub1 = rag.subgraph(nodes1)
    sub2 = rag.subgraph(nodes2)
    return (sub1, sub2)