from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def draw_circular(G, **kwargs):
    """Draw the graph `G` with a circular layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.circular_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called. For
    repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.circular_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.circular_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.draw_circular(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.circular_layout`
    """
    draw(G, circular_layout(G), **kwargs)