from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state
@py_random_state(3)
@nx._dispatch(preserve_edge_attrs=True)
def random_spanning_tree(G, weight=None, *, multiplicative=True, seed=None):
    """
    Sample a random spanning tree using the edges weights of `G`.

    This function supports two different methods for determining the
    probability of the graph. If ``multiplicative=True``, the probability
    is based on the product of edge weights, and if ``multiplicative=False``
    it is based on the sum of the edge weight. However, since it is
    easier to determine the total weight of all spanning trees for the
    multiplicative version, that is significantly faster and should be used if
    possible. Additionally, setting `weight` to `None` will cause a spanning tree
    to be selected with uniform probability.

    The function uses algorithm A8 in [1]_ .

    Parameters
    ----------
    G : nx.Graph
        An undirected version of the original graph.

    weight : string
        The edge key for the edge attribute holding edge weight.

    multiplicative : bool, default=True
        If `True`, the probability of each tree is the product of its edge weight
        over the sum of the product of all the spanning trees in the graph. If
        `False`, the probability is the sum of its edge weight over the sum of
        the sum of weights for all spanning trees in the graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    nx.Graph
        A spanning tree using the distribution defined by the weight of the tree.

    References
    ----------
    .. [1] V. Kulkarni, Generating random combinatorial objects, Journal of
       Algorithms, 11 (1990), pp. 185–207
    """

    def find_node(merged_nodes, node):
        """
        We can think of clusters of contracted nodes as having one
        representative in the graph. Each node which is not in merged_nodes
        is still its own representative. Since a representative can be later
        contracted, we need to recursively search though the dict to find
        the final representative, but once we know it we can use path
        compression to speed up the access of the representative for next time.

        This cannot be replaced by the standard NetworkX union_find since that
        data structure will merge nodes with less representing nodes into the
        one with more representing nodes but this function requires we merge
        them using the order that contract_edges contracts using.

        Parameters
        ----------
        merged_nodes : dict
            The dict storing the mapping from node to representative
        node
            The node whose representative we seek

        Returns
        -------
        The representative of the `node`
        """
        if node not in merged_nodes:
            return node
        else:
            rep = find_node(merged_nodes, merged_nodes[node])
            merged_nodes[node] = rep
            return rep

    def prepare_graph():
        """
        For the graph `G`, remove all edges not in the set `V` and then
        contract all edges in the set `U`.

        Returns
        -------
        A copy of `G` which has had all edges not in `V` removed and all edges
        in `U` contracted.
        """
        result = nx.MultiGraph(incoming_graph_data=G)
        edges_to_remove = set(result.edges()).difference(V)
        result.remove_edges_from(edges_to_remove)
        merged_nodes = {}
        for u, v in U:
            u_rep = find_node(merged_nodes, u)
            v_rep = find_node(merged_nodes, v)
            if u_rep == v_rep:
                continue
            nx.contracted_nodes(result, u_rep, v_rep, self_loops=False, copy=False)
            merged_nodes[v_rep] = u_rep
        return (merged_nodes, result)

    def spanning_tree_total_weight(G, weight):
        """
        Find the sum of weights of the spanning trees of `G` using the
        appropriate `method`.

        This is easy if the chosen method is 'multiplicative', since we can
        use Kirchhoff's Tree Matrix Theorem directly. However, with the
        'additive' method, this process is slightly more complex and less
        computationally efficient as we have to find the number of spanning
        trees which contain each possible edge in the graph.

        Parameters
        ----------
        G : NetworkX Graph
            The graph to find the total weight of all spanning trees on.

        weight : string
            The key for the weight edge attribute of the graph.

        Returns
        -------
        float
            The sum of either the multiplicative or additive weight for all
            spanning trees in the graph.
        """
        if multiplicative:
            return nx.total_spanning_tree_weight(G, weight)
        elif G.number_of_edges() == 1:
            return G.edges(data=weight).__iter__().__next__()[2]
        else:
            total = 0
            for u, v, w in G.edges(data=weight):
                total += w * nx.total_spanning_tree_weight(nx.contracted_edge(G, edge=(u, v), self_loops=False), None)
            return total
    U = set()
    st_cached_value = 0
    V = set(G.edges())
    shuffled_edges = list(G.edges())
    seed.shuffle(shuffled_edges)
    for u, v in shuffled_edges:
        e_weight = G[u][v][weight] if weight is not None else 1
        node_map, prepared_G = prepare_graph()
        G_total_tree_weight = spanning_tree_total_weight(prepared_G, weight)
        rep_edge = (find_node(node_map, u), find_node(node_map, v))
        if rep_edge in prepared_G.edges:
            prepared_G_e = nx.contracted_edge(prepared_G, edge=rep_edge, self_loops=False)
            G_e_total_tree_weight = spanning_tree_total_weight(prepared_G_e, weight)
            if multiplicative:
                threshold = e_weight * G_e_total_tree_weight / G_total_tree_weight
            else:
                numerator = (st_cached_value + e_weight) * nx.total_spanning_tree_weight(prepared_G_e) + G_e_total_tree_weight
                denominator = st_cached_value * nx.total_spanning_tree_weight(prepared_G) + G_total_tree_weight
                threshold = numerator / denominator
        else:
            threshold = 0.0
        z = seed.uniform(0.0, 1.0)
        if z > threshold:
            V.remove((u, v))
        else:
            st_cached_value += e_weight
            U.add((u, v))
        if len(U) == G.number_of_nodes() - 1:
            spanning_tree = nx.Graph()
            spanning_tree.add_edges_from(U)
            return spanning_tree
    raise Exception(f'Something went wrong! Only {len(U)} edges in the spanning tree!')