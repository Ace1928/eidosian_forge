important in operations research and theoretical computer science.
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
@nx._dispatch(edge_attrs='weight')
def held_karp_ascent(G, weight='weight'):
    """
    Minimizes the Held-Karp relaxation of the TSP for `G`

    Solves the Held-Karp relaxation of the input complete digraph and scales
    the output solution for use in the Asadpour [1]_ ASTP algorithm.

    The Held-Karp relaxation defines the lower bound for solutions to the
    ATSP, although it does return a fractional solution. This is used in the
    Asadpour algorithm as an initial solution which is later rounded to a
    integral tree within the spanning tree polytopes. This function solves
    the relaxation with the branch and bound method in [2]_.

    Parameters
    ----------
    G : nx.DiGraph
        The graph should be a complete weighted directed graph.
        The distance between all paris of nodes should be included.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    Returns
    -------
    OPT : float
        The cost for the optimal solution to the Held-Karp relaxation
    z : dict or nx.Graph
        A symmetrized and scaled version of the optimal solution to the
        Held-Karp relaxation for use in the Asadpour algorithm.

        If an integral solution is found, then that is an optimal solution for
        the ATSP problem and that is returned instead.

    References
    ----------
    .. [1] A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi,
       An o(log n/log log n)-approximation algorithm for the asymmetric
       traveling salesman problem, Operations research, 65 (2017),
       pp. 1043–1061

    .. [2] M. Held, R. M. Karp, The traveling-salesman problem and minimum
           spanning trees, Operations Research, 1970-11-01, Vol. 18 (6),
           pp.1138-1162
    """
    import numpy as np
    from scipy import optimize

    def k_pi():
        """
        Find the set of minimum 1-Arborescences for G at point pi.

        Returns
        -------
        Set
            The set of minimum 1-Arborescences
        """
        G_1 = G.copy()
        minimum_1_arborescences = set()
        minimum_1_arborescence_weight = math.inf
        n = next(G.__iter__())
        G_1.remove_node(n)
        min_root = {'node': None, weight: math.inf}
        max_root = {'node': None, weight: -math.inf}
        for u, v, d in G.edges(n, data=True):
            if d[weight] < min_root[weight]:
                min_root = {'node': v, weight: d[weight]}
            if d[weight] > max_root[weight]:
                max_root = {'node': v, weight: d[weight]}
        min_in_edge = min(G.in_edges(n, data=True), key=lambda x: x[2][weight])
        min_root[weight] = min_root[weight] + min_in_edge[2][weight]
        max_root[weight] = max_root[weight] + min_in_edge[2][weight]
        min_arb_weight = math.inf
        for arb in nx.ArborescenceIterator(G_1):
            arb_weight = arb.size(weight)
            if min_arb_weight == math.inf:
                min_arb_weight = arb_weight
            elif arb_weight > min_arb_weight + max_root[weight] - min_root[weight]:
                break
            for N, deg in arb.in_degree:
                if deg == 0:
                    arb.add_edge(n, N, **{weight: G[n][N][weight]})
                    arb_weight += G[n][N][weight]
                    break
            edge_data = G[N][n]
            G.remove_edge(N, n)
            min_weight = min(G.in_edges(n, data=weight), key=lambda x: x[2])[2]
            min_edges = [(u, v, d) for u, v, d in G.in_edges(n, data=weight) if d == min_weight]
            for u, v, d in min_edges:
                new_arb = arb.copy()
                new_arb.add_edge(u, v, **{weight: d})
                new_arb_weight = arb_weight + d
                if new_arb_weight < minimum_1_arborescence_weight:
                    minimum_1_arborescences.clear()
                    minimum_1_arborescence_weight = new_arb_weight
                if new_arb_weight == minimum_1_arborescence_weight:
                    minimum_1_arborescences.add(new_arb)
            G.add_edge(N, n, **edge_data)
        return minimum_1_arborescences

    def direction_of_ascent():
        """
        Find the direction of ascent at point pi.

        See [1]_ for more information.

        Returns
        -------
        dict
            A mapping from the nodes of the graph which represents the direction
            of ascent.

        References
        ----------
        .. [1] M. Held, R. M. Karp, The traveling-salesman problem and minimum
           spanning trees, Operations Research, 1970-11-01, Vol. 18 (6),
           pp.1138-1162
        """
        d = {}
        for n in G:
            d[n] = 0
        del n
        minimum_1_arborescences = k_pi()
        while True:
            min_k_d_weight = math.inf
            min_k_d = None
            for arborescence in minimum_1_arborescences:
                weighted_cost = 0
                for n, deg in arborescence.degree:
                    weighted_cost += d[n] * (deg - 2)
                if weighted_cost < min_k_d_weight:
                    min_k_d_weight = weighted_cost
                    min_k_d = arborescence
            if min_k_d_weight > 0:
                return (d, min_k_d)
            for n, deg in min_k_d.degree:
                d[n] += deg - 2
            c = np.full(len(minimum_1_arborescences), -1, dtype=int)
            a_eq = np.empty((len(G) + 1, len(minimum_1_arborescences)), dtype=int)
            b_eq = np.zeros(len(G) + 1, dtype=int)
            b_eq[len(G)] = 1
            for arb_count, arborescence in enumerate(minimum_1_arborescences):
                n_count = len(G) - 1
                for n, deg in arborescence.degree:
                    a_eq[n_count][arb_count] = deg - 2
                    n_count -= 1
                a_eq[len(G)][arb_count] = 1
            program_result = optimize.linprog(c, A_eq=a_eq, b_eq=b_eq)
            if program_result.success:
                return (None, minimum_1_arborescences)

    def find_epsilon(k, d):
        """
        Given the direction of ascent at pi, find the maximum distance we can go
        in that direction.

        Parameters
        ----------
        k_xy : set
            The set of 1-arborescences which have the minimum rate of increase
            in the direction of ascent

        d : dict
            The direction of ascent

        Returns
        -------
        float
            The distance we can travel in direction `d`
        """
        min_epsilon = math.inf
        for e_u, e_v, e_w in G.edges(data=weight):
            if (e_u, e_v) in k.edges:
                continue
            if len(k.in_edges(e_v, data=weight)) > 1:
                raise Exception
            sub_u, sub_v, sub_w = next(k.in_edges(e_v, data=weight).__iter__())
            k.add_edge(e_u, e_v, **{weight: e_w})
            k.remove_edge(sub_u, sub_v)
            if max((d for n, d in k.in_degree())) <= 1 and len(G) == k.number_of_edges() and nx.is_weakly_connected(k):
                if d[sub_u] == d[e_u] or sub_w == e_w:
                    k.remove_edge(e_u, e_v)
                    k.add_edge(sub_u, sub_v, **{weight: sub_w})
                    continue
                epsilon = (sub_w - e_w) / (d[e_u] - d[sub_u])
                if 0 < epsilon < min_epsilon:
                    min_epsilon = epsilon
            k.remove_edge(e_u, e_v)
            k.add_edge(sub_u, sub_v, **{weight: sub_w})
        return min_epsilon
    pi_dict = {}
    for n in G:
        pi_dict[n] = 0
    del n
    original_edge_weights = {}
    for u, v, d in G.edges(data=True):
        original_edge_weights[u, v] = d[weight]
    dir_ascent, k_d = direction_of_ascent()
    while dir_ascent is not None:
        max_distance = find_epsilon(k_d, dir_ascent)
        for n, v in dir_ascent.items():
            pi_dict[n] += max_distance * v
        for u, v, d in G.edges(data=True):
            d[weight] = original_edge_weights[u, v] + pi_dict[u]
        dir_ascent, k_d = direction_of_ascent()
    k_max = k_d
    for k in k_max:
        if len([n for n in k if k.degree(n) == 2]) == G.order():
            return (k.size(weight), k)
    x_star = {}
    size_k_max = len(k_max)
    for u, v, d in G.edges(data=True):
        edge_count = 0
        d[weight] = original_edge_weights[u, v]
        for k in k_max:
            if (u, v) in k.edges():
                edge_count += 1
                k[u][v][weight] = original_edge_weights[u, v]
        x_star[u, v] = edge_count / size_k_max
    z_star = {}
    scale_factor = (G.order() - 1) / G.order()
    for u, v in x_star:
        frequency = x_star[u, v] + x_star[v, u]
        if frequency > 0:
            z_star[u, v] = scale_factor * frequency
    del x_star
    return (next(k_max.__iter__()).size(weight), z_star)