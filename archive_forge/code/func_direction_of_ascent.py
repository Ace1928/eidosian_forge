important in operations research and theoretical computer science.
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
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