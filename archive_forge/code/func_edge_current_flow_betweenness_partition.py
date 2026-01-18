import functools
import networkx as nx
@nx._dispatchable(edge_attrs='weight')
def edge_current_flow_betweenness_partition(G, number_of_sets, *, weight=None):
    """Partition created by removing the highest edge current flow betweenness edge.

    This algorithm works by calculating the edge current flow
    betweenness for all edges and removing the edge with the
    highest value. It is then determined whether the graph has
    been broken into at least `number_of_sets` connected
    components. If not the process is repeated.

    Parameters
    ----------
    G : NetworkX Graph, DiGraph or MultiGraph
      Graph to be partitioned

    number_of_sets : int
      Number of sets in the desired partition of the graph

    weight : key, optional (default=None)
      The edge attribute key to use as weights for
      edge current flow betweenness calculations

    Returns
    -------
    C : list of sets
      Partition of G

    Raises
    ------
    NetworkXError
      If number_of_sets is <= 0 or number_of_sets > len(G)

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> part = nx.community.edge_current_flow_betweenness_partition(G, 2)
    >>> {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19, 21} in part
    True
    >>> {8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33} in part
    True


    See Also
    --------
    edge_betweenness_partition

    Notes
    -----
    This algorithm is extremely slow, as the recalculation of the edge
    current flow betweenness is extremely slow.

    References
    ----------
    .. [1] Santo Fortunato 'Community Detection in Graphs' Physical Reports
       Volume 486, Issue 3-5 p. 75-174
       http://arxiv.org/abs/0906.0612
    """
    if number_of_sets <= 0:
        raise nx.NetworkXError('number_of_sets must be >0')
    elif number_of_sets == 1:
        return [set(G)]
    elif number_of_sets == len(G):
        return [{n} for n in G]
    elif number_of_sets > len(G):
        raise nx.NetworkXError('number_of_sets must be <= len(G)')
    rank = functools.partial(nx.edge_current_flow_betweenness_centrality, normalized=False, weight=weight)
    H = G.copy()
    partition = list(nx.connected_components(H))
    if len(partition) > 1:
        Hcc_subgraphs = [H.subgraph(cc).copy() for cc in partition]
    else:
        Hcc_subgraphs = [H]
    ranking = {}
    for Hcc in Hcc_subgraphs:
        ranking.update(rank(Hcc))
    while len(partition) < number_of_sets:
        edge = max(ranking, key=ranking.get)
        for cc, Hcc in zip(partition, Hcc_subgraphs):
            if edge[0] in cc:
                Hcc.remove_edge(*edge)
                del ranking[edge]
                splitcc_list = list(nx.connected_components(Hcc))
                if len(splitcc_list) > 1:
                    cc_new = min(splitcc_list, key=len)
                    Hcc_new = Hcc.subgraph(cc_new).copy()
                    newranks = rank(Hcc_new)
                    for e, r in newranks.items():
                        ranking[e if e in ranking else e[::-1]] = r
                    partition.append(cc_new)
                    Hcc_subgraphs.append(Hcc_new)
                    Hcc.remove_nodes_from(cc_new)
                    cc.difference_update(cc_new)
                newranks = rank(Hcc)
                for e, r in newranks.items():
                    ranking[e if e in ranking else e[::-1]] = r
                break
    return partition