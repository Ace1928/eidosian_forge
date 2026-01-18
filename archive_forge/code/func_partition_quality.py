from itertools import combinations
import networkx as nx
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils.decorators import argmap
@require_partition
@nx._dispatch
def partition_quality(G, partition):
    """Returns the coverage and performance of a partition of G.

    The *coverage* of a partition is the ratio of the number of
    intra-community edges to the total number of edges in the graph.

    The *performance* of a partition is the number of
    intra-community edges plus inter-community non-edges divided by the total
    number of potential edges.

    This algorithm has complexity $O(C^2 + L)$ where C is the number of communities and L is the number of links.

    Parameters
    ----------
    G : NetworkX graph

    partition : sequence
        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes (blocks). Each block of the partition represents a
        community.

    Returns
    -------
    (float, float)
        The (coverage, performance) tuple of the partition, as defined above.

    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.

    Notes
    -----
    If `G` is a multigraph;
        - for coverage, the multiplicity of edges is counted
        - for performance, the result is -1 (total number of possible edges is not defined)

    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <https://arxiv.org/abs/0906.0612>
    """
    node_community = {}
    for i, community in enumerate(partition):
        for node in community:
            node_community[node] = i
    if not G.is_multigraph():
        possible_inter_community_edges = sum((len(p1) * len(p2) for p1, p2 in combinations(partition, 2)))
        if G.is_directed():
            possible_inter_community_edges *= 2
    else:
        possible_inter_community_edges = 0
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2
    intra_community_edges = 0
    inter_community_non_edges = possible_inter_community_edges
    for e in G.edges():
        if node_community[e[0]] == node_community[e[1]]:
            intra_community_edges += 1
        else:
            inter_community_non_edges -= 1
    coverage = intra_community_edges / len(G.edges)
    if G.is_multigraph():
        performance = -1.0
    else:
        performance = (intra_community_edges + inter_community_non_edges) / total_pairs
    return (coverage, performance)