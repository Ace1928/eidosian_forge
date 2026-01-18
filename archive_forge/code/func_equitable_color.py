from collections import defaultdict
import networkx as nx
@nx._dispatch
def equitable_color(G, num_colors):
    """Provides an equitable coloring for nodes of `G`.

    Attempts to color a graph using `num_colors` colors, where no neighbors of
    a node can have same color as the node itself and the number of nodes with
    each color differ by at most 1. `num_colors` must be greater than the
    maximum degree of `G`. The algorithm is described in [1]_ and has
    complexity O(num_colors * n**2).

    Parameters
    ----------
    G : networkX graph
       The nodes of this graph will be colored.

    num_colors : number of colors to use
       This number must be at least one more than the maximum degree of nodes
       in the graph.

    Returns
    -------
    A dictionary with keys representing nodes and values representing
    corresponding coloring.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> nx.coloring.equitable_color(G, num_colors=3)  # doctest: +SKIP
    {0: 2, 1: 1, 2: 2, 3: 0}

    Raises
    ------
    NetworkXAlgorithmError
        If `num_colors` is not at least the maximum degree of the graph `G`

    References
    ----------
    .. [1] Kierstead, H. A., Kostochka, A. V., Mydlarz, M., & Szemerédi, E.
        (2010). A fast algorithm for equitable coloring. Combinatorica, 30(2),
        217-224.
    """
    nodes_to_int = {}
    int_to_nodes = {}
    for idx, node in enumerate(G.nodes):
        nodes_to_int[node] = idx
        int_to_nodes[idx] = node
    G = nx.relabel_nodes(G, nodes_to_int, copy=True)
    if len(G.nodes) > 0:
        r_ = max((G.degree(node) for node in G.nodes))
    else:
        r_ = 0
    if r_ >= num_colors:
        raise nx.NetworkXAlgorithmError(f'Graph has maximum degree {r_}, needs {r_ + 1} (> {num_colors}) colors for guaranteed coloring.')
    pad_graph(G, num_colors)
    L_ = {node: [] for node in G.nodes}
    F = {node: idx % num_colors for idx, node in enumerate(G.nodes)}
    C = make_C_from_F(F)
    N = make_N_from_L_C(L_, C)
    H = make_H_from_C_N(C, N)
    edges_seen = set()
    for u in sorted(G.nodes):
        for v in sorted(G.neighbors(u)):
            if (v, u) in edges_seen:
                continue
            edges_seen.add((u, v))
            L_[u].append(v)
            L_[v].append(u)
            N[u, F[v]] += 1
            N[v, F[u]] += 1
            if F[u] != F[v]:
                if N[u, F[v]] == 1:
                    H[F[u], F[v]] -= 1
                if N[v, F[u]] == 1:
                    H[F[v], F[u]] -= 1
        if N[u, F[u]] != 0:
            Y = next((k for k in C if N[u, k] == 0))
            X = F[u]
            change_color(u, X, Y, N=N, H=H, F=F, C=C, L=L_)
            procedure_P(V_minus=X, V_plus=Y, N=N, H=H, F=F, C=C, L=L_)
    return {int_to_nodes[x]: F[x] for x in int_to_nodes}