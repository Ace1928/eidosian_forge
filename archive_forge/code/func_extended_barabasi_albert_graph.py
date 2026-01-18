import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(4)
@nx._dispatch(graphs=None)
def extended_barabasi_albert_graph(n, m, p, q, seed=None):
    """Returns an extended Barabási–Albert model graph.

    An extended Barabási–Albert model graph is a random graph constructed
    using preferential attachment. The extended model allows new edges,
    rewired edges or new nodes. Based on the probabilities $p$ and $q$
    with $p + q < 1$, the growing behavior of the graph is determined as:

    1) With $p$ probability, $m$ new edges are added to the graph,
    starting from randomly chosen existing nodes and attached preferentially at the other end.

    2) With $q$ probability, $m$ existing edges are rewired
    by randomly choosing an edge and rewiring one end to a preferentially chosen node.

    3) With $(1 - p - q)$ probability, $m$ new nodes are added to the graph
    with edges attached preferentially.

    When $p = q = 0$, the model behaves just like the Barabási–Alber model.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges with which a new node attaches to existing nodes
    p : float
        Probability value for adding an edge between existing nodes. p + q < 1
    q : float
        Probability value of rewiring of existing edges. p + q < 1
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n`` or ``1 >= p + q``

    References
    ----------
    .. [1] Albert, R., & Barabási, A. L. (2000)
       Topology of evolving networks: local events and universality
       Physical review letters, 85(24), 5234.
    """
    if m < 1 or m >= n:
        msg = f'Extended Barabasi-Albert network needs m>=1 and m<n, m={m}, n={n}'
        raise nx.NetworkXError(msg)
    if p + q >= 1:
        msg = f'Extended Barabasi-Albert network needs p + q <= 1, p={p}, q={q}'
        raise nx.NetworkXError(msg)
    G = empty_graph(m)
    attachment_preference = []
    attachment_preference.extend(range(m))
    new_node = m
    while new_node < n:
        a_probability = seed.random()
        clique_degree = len(G) - 1
        clique_size = len(G) * clique_degree / 2
        if a_probability < p and G.size() <= clique_size - m:
            eligible_nodes = [nd for nd, deg in G.degree() if deg < clique_degree]
            for i in range(m):
                src_node = seed.choice(eligible_nodes)
                prohibited_nodes = list(G[src_node])
                prohibited_nodes.append(src_node)
                dest_node = seed.choice([nd for nd in attachment_preference if nd not in prohibited_nodes])
                G.add_edge(src_node, dest_node)
                attachment_preference.append(src_node)
                attachment_preference.append(dest_node)
                if G.degree(src_node) == clique_degree:
                    eligible_nodes.remove(src_node)
                if G.degree(dest_node) == clique_degree and dest_node in eligible_nodes:
                    eligible_nodes.remove(dest_node)
        elif p <= a_probability < p + q and m <= G.size() < clique_size:
            eligible_nodes = [nd for nd, deg in G.degree() if 0 < deg < clique_degree]
            for i in range(m):
                node = seed.choice(eligible_nodes)
                neighbor_nodes = list(G[node])
                src_node = seed.choice(neighbor_nodes)
                neighbor_nodes.append(node)
                dest_node = seed.choice([nd for nd in attachment_preference if nd not in neighbor_nodes])
                G.remove_edge(node, src_node)
                G.add_edge(node, dest_node)
                attachment_preference.remove(src_node)
                attachment_preference.append(dest_node)
                if G.degree(src_node) == 0 and src_node in eligible_nodes:
                    eligible_nodes.remove(src_node)
                if dest_node in eligible_nodes:
                    if G.degree(dest_node) == clique_degree:
                        eligible_nodes.remove(dest_node)
                elif G.degree(dest_node) == 1:
                    eligible_nodes.append(dest_node)
        else:
            targets = _random_subset(attachment_preference, m, seed)
            G.add_edges_from(zip([new_node] * m, targets))
            attachment_preference.extend(targets)
            attachment_preference.extend([new_node] * (m + 1))
            new_node += 1
    return G