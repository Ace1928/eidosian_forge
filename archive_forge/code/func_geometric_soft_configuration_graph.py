import math
from bisect import bisect_left
from itertools import accumulate, combinations, product
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(5)
@nx._dispatchable(graphs=None, returns_graph=True)
def geometric_soft_configuration_graph(*, beta, n=None, gamma=None, mean_degree=None, kappas=None, seed=None):
    """Returns a random graph from the geometric soft configuration model.

    The $\\mathbb{S}^1$ model [1]_ is the geometric soft configuration model
    which is able to explain many fundamental features of real networks such as
    small-world property, heteregenous degree distributions, high level of
    clustering, and self-similarity.

    In the geometric soft configuration model, a node $i$ is assigned two hidden
    variables: a hidden degree $\\kappa_i$, quantifying its popularity, influence,
    or importance, and an angular position $\\theta_i$ in a circle abstracting the
    similarity space, where angular distances between nodes are a proxy for their
    similarity. Focusing on the angular position, this model is often called
    the $\\mathbb{S}^1$ model (a one-dimensional sphere). The circle's radius is
    adjusted to $R = N/2\\pi$, where $N$ is the number of nodes, so that the density
    is set to 1 without loss of generality.

    The connection probability between any pair of nodes increases with
    the product of their hidden degrees (i.e., their combined popularities),
    and decreases with the angular distance between the two nodes.
    Specifically, nodes $i$ and $j$ are connected with the probability

    $p_{ij} = \\frac{1}{1 + \\frac{d_{ij}^\\beta}{\\left(\\mu \\kappa_i \\kappa_j\\right)^{\\max(1, \\beta)}}}$

    where $d_{ij} = R\\Delta\\theta_{ij}$ is the arc length of the circle between
    nodes $i$ and $j$ separated by an angular distance $\\Delta\\theta_{ij}$.
    Parameters $\\mu$ and $\\beta$ (also called inverse temperature) control the
    average degree and the clustering coefficient, respectively.

    It can be shown [2]_ that the model undergoes a structural phase transition
    at $\\beta=1$ so that for $\\beta<1$ networks are unclustered in the thermodynamic
    limit (when $N\\to \\infty$) whereas for $\\beta>1$ the ensemble generates
    networks with finite clustering coefficient.

    The $\\mathbb{S}^1$ model can be expressed as a purely geometric model
    $\\mathbb{H}^2$ in the hyperbolic plane [3]_ by mapping the hidden degree of
    each node into a radial coordinate as

    $r_i = \\hat{R} - \\frac{2 \\max(1, \\beta)}{\\beta \\zeta} \\ln \\left(\\frac{\\kappa_i}{\\kappa_0}\\right)$

    where $\\hat{R}$ is the radius of the hyperbolic disk and $\\zeta$ is the curvature,

    $\\hat{R} = \\frac{2}{\\zeta} \\ln \\left(\\frac{N}{\\pi}\\right)
    - \\frac{2\\max(1, \\beta)}{\\beta \\zeta} \\ln (\\mu \\kappa_0^2)$

    The connection probability then reads

    $p_{ij} = \\frac{1}{1 + \\exp\\left({\\frac{\\beta\\zeta}{2} (x_{ij} - \\hat{R})}\\right)}$

    where

    $x_{ij} = r_i + r_j + \\frac{2}{\\zeta} \\ln \\frac{\\Delta\\theta_{ij}}{2}$

    is a good approximation of the hyperbolic distance between two nodes separated
    by an angular distance $\\Delta\\theta_{ij}$ with radial coordinates $r_i$ and $r_j$.
    For $\\beta > 1$, the curvature $\\zeta = 1$, for $\\beta < 1$, $\\zeta = \\beta^{-1}$.


    Parameters
    ----------
    Either `n`, `gamma`, `mean_degree` are provided or `kappas`. The values of
    `n`, `gamma`, `mean_degree` (if provided) are used to construct a random
    kappa-dict keyed by node with values sampled from a power-law distribution.

    beta : positive number
        Inverse temperature, controlling the clustering coefficient.
    n : int (default: None)
        Size of the network (number of nodes).
        If not provided, `kappas` must be provided and holds the nodes.
    gamma : float (default: None)
        Exponent of the power-law distribution for hidden degrees `kappas`.
        If not provided, `kappas` must be provided directly.
    mean_degree : float (default: None)
        The mean degree in the network.
        If not provided, `kappas` must be provided directly.
    kappas : dict (default: None)
        A dict keyed by node to its hidden degree value.
        If not provided, random values are computed based on a power-law
        distribution using `n`, `gamma` and `mean_degree`.
    seed : int, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    Graph
        A random geometric soft configuration graph (undirected with no self-loops).
        Each node has three node-attributes:

        - ``kappa`` that represents the hidden degree.

        - ``theta`` the position in the similarity space ($\\mathbb{S}^1$) which is
          also the angular position in the hyperbolic plane.

        - ``radius`` the radial position in the hyperbolic plane
          (based on the hidden degree).


    Examples
    --------
    Generate a network with specified parameters:

    >>> G = nx.geometric_soft_configuration_graph(beta=1.5, n=100, gamma=2.7, mean_degree=5)

    Create a geometric soft configuration graph with 100 nodes. The $\\beta$ parameter
    is set to 1.5 and the exponent of the powerlaw distribution of the hidden
    degrees is 2.7 with mean value of 5.

    Generate a network with predefined hidden degrees:

    >>> kappas = {i: 10 for i in range(100)}
    >>> G = nx.geometric_soft_configuration_graph(beta=2.5, kappas=kappas)

    Create a geometric soft configuration graph with 100 nodes. The $\\beta$ parameter
    is set to 2.5 and all nodes with hidden degree $\\kappa=10$.


    References
    ----------
    .. [1] Serrano, M. Á., Krioukov, D., & Boguñá, M. (2008). Self-similarity
       of complex networks and hidden metric spaces. Physical review letters, 100(7), 078701.

    .. [2] van der Kolk, J., Serrano, M. Á., & Boguñá, M. (2022). An anomalous
       topological phase transition in spatial random graphs. Communications Physics, 5(1), 245.

    .. [3] Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., & Boguná, M. (2010).
       Hyperbolic geometry of complex networks. Physical Review E, 82(3), 036106.

    """
    if beta <= 0:
        raise nx.NetworkXError('The parameter beta cannot be smaller or equal to 0.')
    if kappas is not None:
        if not all((n is None, gamma is None, mean_degree is None)):
            raise nx.NetworkXError('When kappas is input, n, gamma and mean_degree must not be.')
        n = len(kappas)
        mean_degree = sum(kappas) / len(kappas)
    else:
        if any((n is None, gamma is None, mean_degree is None)):
            raise nx.NetworkXError('Please provide either kappas, or all 3 of: n, gamma and mean_degree.')
        gam_ratio = (gamma - 2) / (gamma - 1)
        kappa_0 = mean_degree * gam_ratio * (1 - 1 / n) / (1 - 1 / n ** gam_ratio)
        base = 1 - 1 / n
        power = 1 / (1 - gamma)
        kappas = {i: kappa_0 * (1 - seed.random() * base) ** power for i in range(n)}
    G = nx.Graph()
    R = n / (2 * math.pi)
    if beta > 1:
        mu = beta * math.sin(math.pi / beta) / (2 * math.pi * mean_degree)
    elif beta == 1:
        mu = 1 / (2 * mean_degree * math.log(n))
    else:
        mu = (1 - beta) / (2 ** beta * mean_degree * n ** (1 - beta))
    thetas = {k: seed.uniform(0, 2 * math.pi) for k in kappas}
    for u in kappas:
        for v in list(G):
            angle = math.pi - math.fabs(math.pi - math.fabs(thetas[u] - thetas[v]))
            dij = math.pow(R * angle, beta)
            mu_kappas = math.pow(mu * kappas[u] * kappas[v], max(1, beta))
            p_ij = 1 / (1 + dij / mu_kappas)
            if seed.random() < p_ij:
                G.add_edge(u, v)
        G.add_node(u)
    nx.set_node_attributes(G, thetas, 'theta')
    nx.set_node_attributes(G, kappas, 'kappa')
    zeta = 1 if beta > 1 else 1 / beta
    kappa_min = min(kappas.values())
    R_c = 2 * max(1, beta) / (beta * zeta)
    R_hat = 2 / zeta * math.log(n / math.pi) - R_c * math.log(mu * kappa_min)
    radii = {node: R_hat - R_c * math.log(kappa) for node, kappa in kappas.items()}
    nx.set_node_attributes(G, radii, 'radius')
    return G