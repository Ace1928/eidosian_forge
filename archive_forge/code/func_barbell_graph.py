import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def barbell_graph(m1, m2, create_using=None):
    """Returns the Barbell Graph: two complete graphs connected by a path.

    Parameters
    ----------
    m1 : int
        Size of the left and right barbells, must be greater than 2.

    m2 : int
        Length of the path connecting the barbells.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
       Only undirected Graphs are supported.

    Returns
    -------
    G : NetworkX graph
        A barbell graph.

    Notes
    -----


    Two identical complete graphs $K_{m1}$ form the left and right bells,
    and are connected by a path $P_{m2}$.

    The `2*m1+m2`  nodes are numbered
        `0, ..., m1-1` for the left barbell,
        `m1, ..., m1+m2-1` for the path,
        and `m1+m2, ..., 2*m1+m2-1` for the right barbell.

    The 3 subgraphs are joined via the edges `(m1-1, m1)` and
    `(m1+m2-1, m1+m2)`. If `m2=0`, this is merely two complete
    graphs joined together.

    This graph is an extremal example in David Aldous
    and Jim Fill's e-text on Random Walks on Graphs.

    """
    if m1 < 2:
        raise NetworkXError('Invalid graph description, m1 should be >=2')
    if m2 < 0:
        raise NetworkXError('Invalid graph description, m2 should be >=0')
    G = complete_graph(m1, create_using)
    if G.is_directed():
        raise NetworkXError('Directed Graph not supported')
    G.add_nodes_from(range(m1, m1 + m2 - 1))
    if m2 > 1:
        G.add_edges_from(pairwise(range(m1, m1 + m2)))
    G.add_edges_from(((u, v) for u in range(m1 + m2, 2 * m1 + m2) for v in range(u + 1, 2 * m1 + m2)))
    G.add_edge(m1 - 1, m1)
    if m2 > 0:
        G.add_edge(m1 + m2 - 1, m1 + m2)
    return G