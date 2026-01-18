import networkx as nx
@nx._dispatch(graphs=None)
def florentine_families_graph():
    """Returns Florentine families graph.

    References
    ----------
    .. [1] Ronald L. Breiger and Philippa E. Pattison
       Cumulated social roles: The duality of persons and their algebras,1
       Social Networks, Volume 8, Issue 3, September 1986, Pages 215-256
    """
    G = nx.Graph()
    G.add_edge('Acciaiuoli', 'Medici')
    G.add_edge('Castellani', 'Peruzzi')
    G.add_edge('Castellani', 'Strozzi')
    G.add_edge('Castellani', 'Barbadori')
    G.add_edge('Medici', 'Barbadori')
    G.add_edge('Medici', 'Ridolfi')
    G.add_edge('Medici', 'Tornabuoni')
    G.add_edge('Medici', 'Albizzi')
    G.add_edge('Medici', 'Salviati')
    G.add_edge('Salviati', 'Pazzi')
    G.add_edge('Peruzzi', 'Strozzi')
    G.add_edge('Peruzzi', 'Bischeri')
    G.add_edge('Strozzi', 'Ridolfi')
    G.add_edge('Strozzi', 'Bischeri')
    G.add_edge('Ridolfi', 'Tornabuoni')
    G.add_edge('Tornabuoni', 'Guadagni')
    G.add_edge('Albizzi', 'Ginori')
    G.add_edge('Albizzi', 'Guadagni')
    G.add_edge('Bischeri', 'Guadagni')
    G.add_edge('Guadagni', 'Lamberteschi')
    return G