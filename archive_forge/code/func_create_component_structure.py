import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def create_component_structure(G):
    """Create component structure for G.

    A *component structure* is an `nxn` array, denoted `c`, where `n` is
    the number of vertices,  where each row and column corresponds to a vertex.

    .. math::
        c_{uv} = \\begin{cases} 0, if v \\in N[u] \\\\
            k, if v \\in component k of G \\setminus N[u] \\end{cases}

    Where `k` is an arbitrary label for each component. The structure is used
    to simplify the detection of asteroidal triples.

    Parameters
    ----------
    G : NetworkX Graph
        Undirected, simple graph.

    Returns
    -------
    component_structure : dictionary
        A dictionary of dictionaries, keyed by pairs of vertices.

    """
    V = set(G.nodes)
    component_structure = {}
    for v in V:
        label = 0
        closed_neighborhood = set(G[v]).union({v})
        row_dict = {}
        for u in closed_neighborhood:
            row_dict[u] = 0
        G_reduced = G.subgraph(set(G.nodes) - closed_neighborhood)
        for cc in nx.connected_components(G_reduced):
            label += 1
            for u in cc:
                row_dict[u] = label
        component_structure[v] = row_dict
    return component_structure