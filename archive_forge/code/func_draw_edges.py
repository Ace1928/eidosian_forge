from collections.abc import Iterable
from itertools import islice, cycle
from numbers import Number
import numpy as np
import rustworkx
def draw_edges(graph, pos, edge_list=None, width=1.0, edge_color='k', style='solid', alpha=None, arrowstyle=None, arrow_size=10, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=True, label=None, node_size=300, node_list=None, node_shape='o', connectionstyle='arc3', min_source_margin=0, min_target_margin=0):
    """Draw the edges of the graph.

    This draws only the edges of the graph.

    Parameters
    ----------
    graph: A rustworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_list : collection of edge tuples (default=graph.edge_list())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edge_list. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax
        parameters.

    style : string (default=solid line)
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or None (default=None)
        The edge transparency

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
        For directed graphs, if True set default to drawing arrowheads.
        Otherwise set default to no arrowheads. Ignored if `arrowstyle` is set.

        Note: Arrows will be the same color as edges.

    arrowstyle : str (default='-\\|>' if directed else '-')
        For directed graphs and `arrows==True` defaults to '-\\|>',
        otherwise defaults to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrow_size : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        ``mutation_scale`` for more info.

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    node_list : list, optional (default=graph.node_indices())
       This provides the node order for the `node_size` array (if it is an
       array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of
        'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int (default=0)
        The minimum margin (gap) at the begining of the edge at the source.

    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.
    """
    try:
        import matplotlib as mpl
        import matplotlib.colors
        import matplotlib.patches
        import matplotlib.path
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running rustworkx.visualization.mpl_draw(). You can install matplotlib with:\n'pip install matplotlib'") from e
    if arrowstyle is None:
        if isinstance(graph, rustworkx.PyDiGraph) and arrows:
            arrowstyle = '-|>'
        else:
            arrowstyle = '-'
    if ax is None:
        ax = plt.gca()
    if edge_list is None:
        edge_list = graph.edge_list()
    if len(edge_list) == 0:
        return []
    if node_list is None:
        node_list = list(graph.node_indices())
    if edge_color is None:
        edge_color = 'k'
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edge_list])
    if np.iterable(edge_color) and len(edge_color) == len(edge_pos) and np.all([isinstance(c, Number) for c in edge_color]):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    def to_marker_edge(marker_size, marker):
        if marker in 's^>v<d':
            return np.sqrt(2 * marker_size) / 2
        else:
            return np.sqrt(marker_size) / 2
    arrow_collection = []
    mutation_scale = arrow_size
    mirustworkx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - mirustworkx
    h = maxy - miny
    base_connectionstyle = mpl.patches.ConnectionStyle(connectionstyle)
    max_nodesize = np.array(node_size).max()

    def _connectionstyle(posA, posB, *args, **kwargs):
        if np.all(posA == posB):
            selfloop_ht = 0.005 * max_nodesize if h == 0 else h
            data_loc = ax.transData.inverted().transform(posA)
            v_shift = 0.1 * selfloop_ht
            h_shift = v_shift * 0.5
            path = [data_loc + np.asarray([0, v_shift]), data_loc + np.asarray([h_shift, v_shift]), data_loc + np.asarray([h_shift, 0]), data_loc, data_loc + np.asarray([-h_shift, 0]), data_loc + np.asarray([-h_shift, v_shift]), data_loc + np.asarray([0, v_shift])]
            ret = mpl.path.Path(ax.transData.transform(path), [1, 4, 4, 4, 4, 4, 4])
        else:
            ret = base_connectionstyle(posA, posB, *args, **kwargs)
        return ret
    arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
    for i, (src, dst) in enumerate(edge_pos):
        x1, y1 = src
        x2, y2 = dst
        shrink_source = 0
        shrink_target = 0
        if np.iterable(node_size):
            source, target = edge_list[i][:2]
            source_node_size = node_size[node_list.index(source)]
            target_node_size = node_size[node_list.index(target)]
            shrink_source = to_marker_edge(source_node_size, node_shape)
            shrink_target = to_marker_edge(target_node_size, node_shape)
        else:
            shrink_source = shrink_target = to_marker_edge(node_size, node_shape)
        if shrink_source < min_source_margin:
            shrink_source = min_source_margin
        if shrink_target < min_target_margin:
            shrink_target = min_target_margin
        if len(arrow_colors) == len(edge_pos):
            arrow_color = arrow_colors[i]
        elif len(arrow_colors) == 1:
            arrow_color = arrow_colors[0]
        else:
            arrow_color = arrow_colors[i % len(arrow_colors)]
        if np.iterable(width):
            if len(width) == len(edge_pos):
                line_width = width[i]
            else:
                line_width = width[i % len(width)]
        else:
            line_width = width
        arrow = mpl.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=arrowstyle, shrinkA=shrink_source, shrinkB=shrink_target, mutation_scale=mutation_scale, color=arrow_color, linewidth=line_width, connectionstyle=_connectionstyle, linestyle=style, zorder=1)
        arrow_collection.append(arrow)
        ax.add_patch(arrow)
    padx, pady = (0.05 * w, 0.05 * h)
    corners = ((mirustworkx - padx, miny - pady), (maxx + padx, maxy + pady))
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return arrow_collection