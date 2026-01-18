from collections.abc import Iterable
from itertools import islice, cycle
from numbers import Number
import numpy as np
import rustworkx
def draw_labels(graph, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, clip_on=True):
    """Draw node labels on the graph.

    Parameters
    ----------
    graph: A rustworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    labels : dictionary (default={n: n for n in graph})
        Node labels in a dictionary of text labels keyed by node.
        Node-keys in labels should appear as keys in `pos`.
        If needed use: `{n:lab for n,lab in labels.items() if n in pos}`

    font_size : int (default=12)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)
        Specify text box properties (e.g. shape, color etc.) for node labels.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline',
                            'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    clip_on : bool (default=True)
        Turn on clipping of node labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed on the nodes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running rustworkx.visualization.mpl_draw(). You can install matplotlib with:\n'pip install matplotlib'") from e
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = {n: n for n in graph.node_indices()}
    text_items = {}
    for n, label in labels.items():
        x, y = pos[n]
        if not isinstance(label, str):
            label = str(label)
        t = ax.text(x, y, label, size=font_size, color=font_color, family=font_family, weight=font_weight, alpha=alpha, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, transform=ax.transData, bbox=bbox, clip_on=clip_on)
        text_items[n] = t
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return text_items