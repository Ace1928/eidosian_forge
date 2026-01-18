import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
def merge_nodes(self, src, dst, weight_func=min_weight, in_place=True, extra_arguments=None, extra_keywords=None):
    """Merge node `src` and `dst`.

        The new combined node is adjacent to all the neighbors of `src`
        and `dst`. `weight_func` is called to decide the weight of edges
        incident on the new node.

        Parameters
        ----------
        src, dst : int
            Nodes to be merged.
        weight_func : callable, optional
            Function to decide the attributes of edges incident on the new
            node. For each neighbor `n` for `src` and `dst`, `weight_func` will
            be called as follows: `weight_func(src, dst, n, *extra_arguments,
            **extra_keywords)`. `src`, `dst` and `n` are IDs of vertices in the
            RAG object which is in turn a subclass of :obj:`networkx.Graph`. It is
            expected to return a dict of attributes of the resulting edge.
        in_place : bool, optional
            If set to `True`, the merged node has the id `dst`, else merged
            node has a new id which is returned.
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `weight_func`.
        extra_keywords : dictionary, optional
            The dict of keyword arguments passed to the `weight_func`.

        Returns
        -------
        id : int
            The id of the new node.

        Notes
        -----
        If `in_place` is `False` the resulting node has a new id, rather than
        `dst`.
        """
    if extra_arguments is None:
        extra_arguments = []
    if extra_keywords is None:
        extra_keywords = {}
    src_nbrs = set(self.neighbors(src))
    dst_nbrs = set(self.neighbors(dst))
    neighbors = (src_nbrs | dst_nbrs) - {src, dst}
    if in_place:
        new = dst
    else:
        new = self.next_id()
        self.add_node(new)
    for neighbor in neighbors:
        data = weight_func(self, src, dst, neighbor, *extra_arguments, **extra_keywords)
        self.add_edge(neighbor, new, attr_dict=data)
    self.nodes[new]['labels'] = self.nodes[src]['labels'] + self.nodes[dst]['labels']
    self.remove_node(src)
    if not in_place:
        self.remove_node(dst)
    return new