import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def dorogovtsev_goltsev_mendes_graph(n, create_using=None):
    """Returns the hierarchically constructed Dorogovtsev-Goltsev-Mendes graph.

    The Dorogovtsev-Goltsev-Mendes [1]_ procedure produces a scale-free graph
    deterministically with the following properties for a given `n`:
    - Total number of nodes = ``3 * (3**n + 1) / 2``
    - Total number of edges = ``3 ** (n + 1)``

    Parameters
    ----------
    n : integer
       The generation number.

    create_using : NetworkX Graph, optional
       Graph type to be returned. Directed graphs and multi graphs are not
       supported.

    Returns
    -------
    G : NetworkX Graph

    Examples
    --------
    >>> G = nx.dorogovtsev_goltsev_mendes_graph(3)
    >>> G.number_of_nodes()
    15
    >>> G.number_of_edges()
    27
    >>> nx.is_planar(G)
    True

    References
    ----------
    .. [1] S. N. Dorogovtsev, A. V. Goltsev and J. F. F. Mendes,
        "Pseudofractal scale-free web", Physical Review E 65, 066122, 2002.
        https://arxiv.org/pdf/cond-mat/0112143.pdf
    """
    G = empty_graph(0, create_using)
    if G.is_directed():
        raise NetworkXError('Directed Graph not supported')
    if G.is_multigraph():
        raise NetworkXError('Multigraph not supported')
    G.add_edge(0, 1)
    if n == 0:
        return G
    new_node = 2
    for i in range(1, n + 1):
        last_generation_edges = list(G.edges())
        number_of_edges_in_last_generation = len(last_generation_edges)
        for j in range(number_of_edges_in_last_generation):
            G.add_edge(new_node, last_generation_edges[j][0])
            G.add_edge(new_node, last_generation_edges[j][1])
            new_node += 1
    return G