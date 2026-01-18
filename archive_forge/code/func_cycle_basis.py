from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def cycle_basis(G, root=None):
    """Returns a list of cycles which form a basis for cycles of G.

    A basis for cycles of a network is a minimal collection of
    cycles such that any cycle in the network can be written
    as a sum of cycles in the basis.  Here summation of cycles
    is defined as "exclusive or" of the edges. Cycle bases are
    useful, e.g. when deriving equations for electric circuits
    using Kirchhoff's Laws.

    Parameters
    ----------
    G : NetworkX Graph
    root : node, optional
       Specify starting node for basis.

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [0, 3, 4, 5])
    >>> nx.cycle_basis(G, 0)
    [[3, 4, 5, 0], [1, 2, 3, 0]]

    Notes
    -----
    This is adapted from algorithm CACM 491 [1]_.

    References
    ----------
    .. [1] Paton, K. An algorithm for finding a fundamental set of
       cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

    See Also
    --------
    simple_cycles
    """
    gnodes = dict.fromkeys(G)
    cycles = []
    while gnodes:
        if root is None:
            root = gnodes.popitem()[0]
        stack = [root]
        pred = {root: root}
        used = {root: set()}
        while stack:
            z = stack.pop()
            zused = used[z]
            for nbr in G[z]:
                if nbr not in used:
                    pred[nbr] = z
                    stack.append(nbr)
                    used[nbr] = {z}
                elif nbr == z:
                    cycles.append([z])
                elif nbr not in zused:
                    pn = used[nbr]
                    cycle = [nbr, z]
                    p = pred[z]
                    while p not in pn:
                        cycle.append(p)
                        p = pred[p]
                    cycle.append(p)
                    cycles.append(cycle)
                    used[nbr].add(z)
        for node in pred:
            gnodes.pop(node, None)
        root = None
    return cycles