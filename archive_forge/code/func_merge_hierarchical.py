import numpy as np
import heapq
def merge_hierarchical(labels, rag, thresh, rag_copy, in_place_merge, merge_func, weight_func):
    """Perform hierarchical merging of a RAG.

    Greedily merges the most similar pair of nodes until no edges lower than
    `thresh` remain.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    thresh : float
        Regions connected by an edge with weight smaller than `thresh` are
        merged.
    rag_copy : bool
        If set, the RAG copied before modifying.
    in_place_merge : bool
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge..
    merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.

    Returns
    -------
    out : ndarray
        The new labeled array.

    """
    if rag_copy:
        rag = rag.copy()
    edge_heap = []
    for n1, n2, data in rag.edges(data=True):
        wt = data['weight']
        heap_item = [wt, n1, n2, True]
        heapq.heappush(edge_heap, heap_item)
        data['heap item'] = heap_item
    while len(edge_heap) > 0 and edge_heap[0][0] < thresh:
        _, n1, n2, valid = heapq.heappop(edge_heap)
        if valid:
            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)
            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)
            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                src, dst = (n1, next_id)
            else:
                src, dst = (n1, n2)
            merge_func(rag, src, dst)
            new_id = rag.merge_nodes(src, dst, weight_func)
            _revalidate_node_edges(rag, new_id, edge_heap)
    label_map = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for label in d['labels']:
            label_map[label] = ix
    return label_map[labels]