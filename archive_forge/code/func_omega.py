import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatch
def omega(G, niter=5, nrand=10, seed=None):
    """Returns the small-world coefficient (omega) of a graph

    The small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    The small-world coefficient (omega) measures how much G is like a lattice
    or a random graph. Negative values mean G is similar to a lattice whereas
    positive values mean G is a random graph.
    Values close to 0 mean that G has small-world characteristics.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.

    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the maximal clustering
        coefficient (Cr) and average shortest path length (Lr).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.


    Returns
    -------
    omega : float
        The small-world coefficient (omega)

    Notes
    -----
    The implementation is adapted from the algorithm by Telesford et al. [1]_.

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.
    """
    import numpy as np
    randMetrics = {'C': [], 'L': []}
    Cl = nx.average_clustering(G)
    niter_lattice_reference = niter
    niter_random_reference = niter * 2
    for _ in range(nrand):
        Gr = random_reference(G, niter=niter_random_reference, seed=seed)
        randMetrics['L'].append(nx.average_shortest_path_length(Gr))
        Gl = lattice_reference(G, niter=niter_lattice_reference, seed=seed)
        Cl_temp = nx.average_clustering(Gl)
        if Cl_temp > Cl:
            Cl = Cl_temp
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    Lr = np.mean(randMetrics['L'])
    omega = Lr / L - C / Cl
    return omega