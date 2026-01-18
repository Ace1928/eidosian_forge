import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatch
def sigma(G, niter=100, nrand=10, seed=None):
    """Returns the small-world coefficient (sigma) of the given graph.

    The small-world coefficient is defined as:
    sigma = C/Cr / L/Lr
    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Cr and Lr are respectively the average
    clustering coefficient and average shortest path length of an equivalent
    random graph.

    A graph is commonly classified as small-world if sigma>1.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    niter : integer (optional, default=100)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.
    nrand : integer (optional, default=10)
        Number of random graphs generated to compute the average clustering
        coefficient (Cr) and average shortest path length (Lr).
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    sigma : float
        The small-world coefficient of G.

    Notes
    -----
    The implementation is adapted from Humphries et al. [1]_ [2]_.

    References
    ----------
    .. [1] The brainstem reticular formation is a small-world, not scale-free,
           network M. D. Humphries, K. Gurney and T. J. Prescott,
           Proc. Roy. Soc. B 2006 273, 503-511, doi:10.1098/rspb.2005.3354.
    .. [2] Humphries and Gurney (2008).
           "Network 'Small-World-Ness': A Quantitative Method for Determining
           Canonical Network Equivalence".
           PLoS One. 3 (4). PMID 18446219. doi:10.1371/journal.pone.0002051.
    """
    import numpy as np
    randMetrics = {'C': [], 'L': []}
    for i in range(nrand):
        Gr = random_reference(G, niter=niter, seed=seed)
        randMetrics['C'].append(nx.transitivity(Gr))
        randMetrics['L'].append(nx.average_shortest_path_length(Gr))
    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
    Cr = np.mean(randMetrics['C'])
    Lr = np.mean(randMetrics['L'])
    sigma = C / Cr / (L / Lr)
    return sigma