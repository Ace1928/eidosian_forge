import networkx as nx
from networkx.exception import NetworkXError
@nx._dispatch(graphs={'G1': 0, 'G2': 1})
def could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.
    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree, triangle, and number of cliques sequences.
    The triangle sequence contains the number of triangles each node is part of.
    The clique sequence contains for each node the number of maximal cliques
    involving that node.

    """
    if G1.order() != G2.order():
        return False
    d1 = G1.degree()
    t1 = nx.triangles(G1)
    clqs_1 = list(nx.find_cliques(G1))
    c1 = {n: sum((1 for c in clqs_1 if n in c)) for n in G1}
    props1 = [[d, t1[v], c1[v]] for v, d in d1]
    props1.sort()
    d2 = G2.degree()
    t2 = nx.triangles(G2)
    clqs_2 = list(nx.find_cliques(G2))
    c2 = {n: sum((1 for c in clqs_2 if n in c)) for n in G2}
    props2 = [[d, t2[v], c2[v]] for v, d in d2]
    props2.sort()
    if props1 != props2:
        return False
    return True