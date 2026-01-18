import math
from bisect import bisect_left
from itertools import accumulate, combinations, product
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(5)
@nx._dispatch(graphs=None)
def navigable_small_world_graph(n, p=1, q=1, r=2, dim=2, seed=None):
    """Returns a navigable small-world graph.

    A navigable small-world graph is a directed grid with additional long-range
    connections that are chosen randomly.

      [...] we begin with a set of nodes [...] that are identified with the set
      of lattice points in an $n \\times n$ square,
      $\\{(i, j): i \\in \\{1, 2, \\ldots, n\\}, j \\in \\{1, 2, \\ldots, n\\}\\}$,
      and we define the *lattice distance* between two nodes $(i, j)$ and
      $(k, l)$ to be the number of "lattice steps" separating them:
      $d((i, j), (k, l)) = |k - i| + |l - j|$.

      For a universal constant $p >= 1$, the node $u$ has a directed edge to
      every other node within lattice distance $p$---these are its *local
      contacts*. For universal constants $q >= 0$ and $r >= 0$ we also
      construct directed edges from $u$ to $q$ other nodes (the *long-range
      contacts*) using independent random trials; the $i$th directed edge from
      $u$ has endpoint $v$ with probability proportional to $[d(u,v)]^{-r}$.

      -- [1]_

    Parameters
    ----------
    n : int
        The length of one side of the lattice; the number of nodes in
        the graph is therefore $n^2$.
    p : int
        The diameter of short range connections. Each node is joined with every
        other node within this lattice distance.
    q : int
        The number of long-range connections for each node.
    r : float
        Exponent for decaying probability of connections.  The probability of
        connecting to a node at lattice distance $d$ is $1/d^r$.
    dim : int
        Dimension of grid
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    References
    ----------
    .. [1] J. Kleinberg. The small-world phenomenon: An algorithmic
       perspective. Proc. 32nd ACM Symposium on Theory of Computing, 2000.
    """
    if p < 1:
        raise nx.NetworkXException('p must be >= 1')
    if q < 0:
        raise nx.NetworkXException('q must be >= 0')
    if r < 0:
        raise nx.NetworkXException('r must be >= 1')
    G = nx.DiGraph()
    nodes = list(product(range(n), repeat=dim))
    for p1 in nodes:
        probs = [0]
        for p2 in nodes:
            if p1 == p2:
                continue
            d = sum((abs(b - a) for a, b in zip(p1, p2)))
            if d <= p:
                G.add_edge(p1, p2)
            probs.append(d ** (-r))
        cdf = list(accumulate(probs))
        for _ in range(q):
            target = nodes[bisect_left(cdf, seed.uniform(0, cdf[-1]))]
            G.add_edge(p1, target)
    return G