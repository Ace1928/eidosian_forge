important in operations research and theoretical computer science.
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
@nx._dispatch
def spanning_tree_distribution(G, z):
    """
    Find the asadpour exponential distribution of spanning trees.

    Solves the Maximum Entropy Convex Program in the Asadpour algorithm [1]_
    using the approach in section 7 to build an exponential distribution of
    undirected spanning trees.

    This algorithm ensures that the probability of any edge in a spanning
    tree is proportional to the sum of the probabilities of the tress
    containing that edge over the sum of the probabilities of all spanning
    trees of the graph.

    Parameters
    ----------
    G : nx.MultiGraph
        The undirected support graph for the Held Karp relaxation

    z : dict
        The output of `held_karp_ascent()`, a scaled version of the Held-Karp
        solution.

    Returns
    -------
    gamma : dict
        The probability distribution which approximately preserves the marginal
        probabilities of `z`.
    """
    from math import exp
    from math import log as ln

    def q(e):
        """
        The value of q(e) is described in the Asadpour paper is "the
        probability that edge e will be included in a spanning tree T that is
        chosen with probability proportional to exp(gamma(T))" which
        basically means that it is the total probability of the edge appearing
        across the whole distribution.

        Parameters
        ----------
        e : tuple
            The `(u, v)` tuple describing the edge we are interested in

        Returns
        -------
        float
            The probability that a spanning tree chosen according to the
            current values of gamma will include edge `e`.
        """
        for u, v, d in G.edges(data=True):
            d[lambda_key] = exp(gamma[u, v])
        G_Kirchhoff = nx.total_spanning_tree_weight(G, lambda_key)
        G_e = nx.contracted_edge(G, e, self_loops=False)
        G_e_Kirchhoff = nx.total_spanning_tree_weight(G_e, lambda_key)
        return exp(gamma[e[0], e[1]]) * G_e_Kirchhoff / G_Kirchhoff
    gamma = {}
    for u, v, _ in G.edges:
        gamma[u, v] = 0
    EPSILON = 0.2
    lambda_key = "spanning_tree_distribution's secret attribute name for lambda"
    while True:
        in_range_count = 0
        for u, v in gamma:
            e = (u, v)
            q_e = q(e)
            z_e = z[e]
            if q_e > (1 + EPSILON) * z_e:
                delta = ln(q_e * (1 - (1 + EPSILON / 2) * z_e) / ((1 - q_e) * (1 + EPSILON / 2) * z_e))
                gamma[e] -= delta
                new_q_e = q(e)
                desired_q_e = (1 + EPSILON / 2) * z_e
                if round(new_q_e, 8) != round(desired_q_e, 8):
                    raise nx.NetworkXError(f'Unable to modify probability for edge ({u}, {v})')
            else:
                in_range_count += 1
        if in_range_count == len(gamma):
            break
    for _, _, d in G.edges(data=True):
        if lambda_key in d:
            del d[lambda_key]
    return gamma