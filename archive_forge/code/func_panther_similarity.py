import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def panther_similarity(G, source, k=5, path_length=5, c=0.5, delta=0.1, eps=None, weight='weight'):
    """Returns the Panther similarity of nodes in the graph `G` to node ``v``.

    Panther is a similarity metric that says "two objects are considered
    to be similar if they frequently appear on the same paths." [1]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph
    source : node
        Source node for which to find the top `k` similar other nodes
    k : int (default = 5)
        The number of most similar nodes to return
    path_length : int (default = 5)
        How long the randomly generated paths should be (``T`` in [1]_)
    c : float (default = 0.5)
        A universal positive constant used to scale the number
        of sample random paths to generate.
    delta : float (default = 0.1)
        The probability that the similarity $S$ is not an epsilon-approximation to (R, phi),
        where $R$ is the number of random paths and $\\phi$ is the probability
        that an element sampled from a set $A \\subseteq D$, where $D$ is the domain.
    eps : float or None (default = None)
        The error bound. Per [1]_, a good value is ``sqrt(1/|E|)``. Therefore,
        if no value is provided, the recommended computed value will be used.
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.

    Returns
    -------
    similarity : dictionary
        Dictionary of nodes to similarity scores (as floats). Note:
        the self-similarity (i.e., ``v``) will not be included in
        the returned dictionary.

    Examples
    --------
    >>> G = nx.star_graph(10)
    >>> sim = nx.panther_similarity(G, 0)

    References
    ----------
    .. [1] Zhang, J., Tang, J., Ma, C., Tong, H., Jing, Y., & Li, J.
           Panther: Fast top-k similarity search on large networks.
           In Proceedings of the ACM SIGKDD International Conference
           on Knowledge Discovery and Data Mining (Vol. 2015-August, pp. 1445–1454).
           Association for Computing Machinery. https://doi.org/10.1145/2783258.2783267.
    """
    import numpy as np
    num_nodes = G.number_of_nodes()
    if num_nodes < k:
        warnings.warn(f'Number of nodes is {num_nodes}, but requested k is {k}. Setting k to number of nodes.')
        k = num_nodes
    if eps is None:
        eps = np.sqrt(1.0 / G.number_of_edges())
    inv_node_map = {name: index for index, name in enumerate(G.nodes)}
    node_map = np.array(G)
    t_choose_2 = math.comb(path_length, 2)
    sample_size = int(c / eps ** 2 * (np.log2(t_choose_2) + 1 + np.log(1 / delta)))
    index_map = {}
    _ = list(generate_random_paths(G, sample_size, path_length=path_length, index_map=index_map, weight=weight))
    S = np.zeros(num_nodes)
    inv_sample_size = 1 / sample_size
    source_paths = set(index_map[source])
    for node, paths in index_map.items():
        common_paths = source_paths.intersection(paths)
        S[inv_node_map[node]] = len(common_paths) * inv_sample_size
    top_k_unsorted = np.argpartition(S, -k)[-k:]
    top_k_sorted = top_k_unsorted[np.argsort(S[top_k_unsorted])][::-1]
    top_k_sorted_names = (node_map[n] for n in top_k_sorted)
    top_k_with_val = dict(zip(top_k_sorted_names, S[top_k_sorted]))
    top_k_with_val.pop(source, None)
    return top_k_with_val