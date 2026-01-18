import math
from collections import deque
import networkx as nx
@nx._dispatch
def bfs_labeled_edges(G, sources):
    """Iterate over edges in a breadth-first search (BFS) labeled by type.

    We generate triple of the form (*u*, *v*, *d*), where (*u*, *v*) is the
    edge being explored in the breadth-first search and *d* is one of the
    strings 'tree', 'forward', 'level', or 'reverse'.  A 'tree' edge is one in
    which *v* is first discovered and placed into the layer below *u*.  A
    'forward' edge is one in which *u* is on the layer above *v* and *v* has
    already been discovered.  A 'level' edge is one in which both *u* and *v*
    occur on the same layer.  A 'reverse' edge is one in which *u* is on a layer
    below *v*.

    We emit each edge exactly once.  In an undirected graph, 'reverse' edges do
    not occur, because each is discovered either as a 'tree' or 'forward' edge.

    Parameters
    ----------
    G : NetworkX graph
        A graph over which to find the layers using breadth-first search.

    sources : node in `G` or list of nodes in `G`
        Starting nodes for single source or multiple sources breadth-first search

    Yields
    ------
    edges: generator
       A generator of triples (*u*, *v*, *d*) where (*u*, *v*) is the edge being
       explored and *d* is described above.

    Examples
    --------
    >>> G = nx.cycle_graph(4, create_using = nx.DiGraph)
    >>> list(nx.bfs_labeled_edges(G, 0))
    [(0, 1, 'tree'), (1, 2, 'tree'), (2, 3, 'tree'), (3, 0, 'reverse')]
    >>> G = nx.complete_graph(3)
    >>> list(nx.bfs_labeled_edges(G, 0))
    [(0, 1, 'tree'), (0, 2, 'tree'), (1, 2, 'level')]
    >>> list(nx.bfs_labeled_edges(G, [0, 1]))
    [(0, 1, 'level'), (0, 2, 'tree'), (1, 2, 'forward')]
    """
    if sources in G:
        sources = [sources]
    neighbors = G._adj
    directed = G.is_directed()
    visited = set()
    visit = visited.discard if directed else visited.add
    depth = {s: 0 for s in sources}
    queue = deque(depth.items())
    push = queue.append
    pop = queue.popleft
    while queue:
        u, du = pop()
        for v in neighbors[u]:
            if v not in depth:
                depth[v] = dv = du + 1
                push((v, dv))
                yield (u, v, TREE_EDGE)
            else:
                dv = depth[v]
                if du == dv:
                    if v not in visited:
                        yield (u, v, LEVEL_EDGE)
                elif du < dv:
                    yield (u, v, FORWARD_EDGE)
                elif directed:
                    yield (u, v, REVERSE_EDGE)
        visit(u)