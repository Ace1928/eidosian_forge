important in operations research and theoretical computer science.
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
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