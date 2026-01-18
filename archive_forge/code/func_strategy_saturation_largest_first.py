import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
@nx._dispatch
def strategy_saturation_largest_first(G, colors):
    """Iterates over all the nodes of ``G`` in "saturation order" (also
    known as "DSATUR").

    ``G`` is a NetworkX graph. ``colors`` is a dictionary mapping nodes of
    ``G`` to colors, for those nodes that have already been colored.

    """
    distinct_colors = {v: set() for v in G}
    for node, color in colors.items():
        for neighbor in G[node]:
            distinct_colors[neighbor].add(color)
    if len(colors) >= 2:
        for node, color in colors.items():
            if color in distinct_colors[node]:
                raise nx.NetworkXError('Neighboring nodes must have different colors')
    if not colors:
        node = max(G, key=G.degree)
        yield node
        for v in G[node]:
            distinct_colors[v].add(0)
    while len(G) != len(colors):
        for node, color in colors.items():
            for neighbor in G[node]:
                distinct_colors[neighbor].add(color)
        saturation = {v: len(c) for v, c in distinct_colors.items() if v not in colors}
        node = max(saturation, key=lambda v: (saturation[v], G.degree(v)))
        yield node