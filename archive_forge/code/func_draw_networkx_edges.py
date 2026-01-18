from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k', style='solid', alpha=None, arrowstyle=None, arrowsize=10, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=None, label=None, node_size=300, nodelist=None, node_shape='o', connectionstyle='arc3', min_source_margin=0, min_target_margin=0):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edgelist : collection of edge tuples (default=G.edges())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string or array of strings (default='solid')
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        Can be a single style or a sequence of styles with the same
        length as the edge list.
        If less styles than edges are given the styles will cycle.
        If more styles than edges are given the styles will be used sequentially
        and not be exhausted.
        Also, `(offset, onoffseq)` tuples can be used as style instead of a strings.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or array of floats (default=None)
        The edge transparency.  This can be a single alpha value,
        in which case it will be applied to all specified edges. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool or None, optional (default=None)
        If `None`, directed graphs draw arrowheads with
        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges
        via `~matplotlib.collections.LineCollection` for speed.
        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).
        If `False`, draw edges using LineCollection (linear and fast).

        Note: Arrowheads will be the same color as edges.

    arrowstyle : str (default='-\\|>' for directed graphs)
        For directed graphs and `arrows==True` defaults to '-\\|>',
        For undirected graphs default to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        `mutation_scale` for more info.

    connectionstyle : string (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    nodelist : list, optional (default=G.nodes())
       This provides the node order for the `node_size` array (if it is an array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of 'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int (default=0)
        The minimum margin (gap) at the beginning of the edge at the source.

    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
     matplotlib.collections.LineCollection or a list of matplotlib.patches.FancyArrowPatch
        If ``arrows=True``, a list of FancyArrowPatches is returned.
        If ``arrows=False``, a LineCollection is returned.
        If ``arrows=None`` (the default), then a LineCollection is returned if
        `G` is undirected, otherwise returns a list of FancyArrowPatches.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.

    Self-loops are always drawn with `~matplotlib.patches.FancyArrowPatch`
    regardless of the value of `arrows` or whether `G` is directed.
    When ``arrows=False`` or ``arrows=None`` and `G` is undirected, the
    FancyArrowPatches corresponding to the self-loops are not explicitly
    returned. They should instead be accessed via the ``Axes.patches``
    attribute (see examples).

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    The FancyArrowPatches corresponding to self-loops are not always
    returned, but can always be accessed via the ``patches`` attribute of the
    `matplotlib.Axes` object.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> G = nx.Graph([(0, 1), (0, 0)])  # Self-loop at node 0
    >>> edge_collection = nx.draw_networkx_edges(G, pos=nx.circular_layout(G), ax=ax)
    >>> self_loop_fap = ax.patches[0]

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_labels
    draw_networkx_edge_labels

    """
    import matplotlib as mpl
    import matplotlib.collections
    import matplotlib.colors
    import matplotlib.patches
    import matplotlib.path
    import matplotlib.pyplot as plt
    import numpy as np
    use_linecollection = not G.is_directed()
    if arrows in (True, False):
        use_linecollection = not arrows
    if use_linecollection and any([arrowstyle is not None, arrowsize != 10, connectionstyle != 'arc3', min_source_margin != 0, min_target_margin != 0]):
        import warnings
        msg = '\n\nThe {0} keyword argument is not applicable when drawing edges\nwith LineCollection.\n\nTo make this warning go away, either specify `arrows=True` to\nforce FancyArrowPatches or use the default value for {0}.\nNote that using FancyArrowPatches may be slow for large graphs.\n'
        if arrowstyle is not None:
            msg = msg.format('arrowstyle')
        if arrowsize != 10:
            msg = msg.format('arrowsize')
        if connectionstyle != 'arc3':
            msg = msg.format('connectionstyle')
        if min_source_margin != 0:
            msg = msg.format('min_source_margin')
        if min_target_margin != 0:
            msg = msg.format('min_target_margin')
        warnings.warn(msg, category=UserWarning, stacklevel=2)
    if arrowstyle == None:
        if G.is_directed():
            arrowstyle = '-|>'
        else:
            arrowstyle = '-'
    if ax is None:
        ax = plt.gca()
    if edgelist is None:
        edgelist = list(G.edges())
    if len(edgelist) == 0:
        return []
    if nodelist is None:
        nodelist = list(G.nodes())
    if edge_color is None:
        edge_color = 'k'
    edgelist_tuple = list(map(tuple, edgelist))
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
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

    def _draw_networkx_edges_line_collection():
        edge_collection = mpl.collections.LineCollection(edge_pos, colors=edge_color, linewidths=width, antialiaseds=(1,), linestyle=style, alpha=alpha)
        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)
        edge_collection.set_zorder(1)
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)
        return edge_collection

    def _draw_networkx_edges_fancy_arrow_patch():

        def to_marker_edge(marker_size, marker):
            if marker in 's^>v<d':
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2
        arrow_collection = []
        if isinstance(arrowsize, list):
            if len(arrowsize) != len(edge_pos):
                raise ValueError('arrowsize should have the same length as edgelist')
        else:
            mutation_scale = arrowsize
        base_connection_style = mpl.patches.ConnectionStyle(connectionstyle)
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
                ret = base_connection_style(posA, posB, *args, **kwargs)
            return ret
        arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in zip(fancy_edges_indices, edge_pos):
            x1, y1 = src
            x2, y2 = dst
            shrink_source = 0
            shrink_target = 0
            if isinstance(arrowsize, list):
                mutation_scale = arrowsize[i]
            if np.iterable(node_size):
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)
            if shrink_source < min_source_margin:
                shrink_source = min_source_margin
            if shrink_target < min_target_margin:
                shrink_target = min_target_margin
            if len(arrow_colors) > i:
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:
                arrow_color = arrow_colors[i % len(arrow_colors)]
            if np.iterable(width):
                if len(width) > i:
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width
            if np.iterable(style) and (not isinstance(style, str)) and (not isinstance(style, tuple)):
                if len(style) > i:
                    linestyle = style[i]
                else:
                    linestyle = style[i % len(style)]
            else:
                linestyle = style
            arrow = mpl.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=arrowstyle, shrinkA=shrink_source, shrinkB=shrink_target, mutation_scale=mutation_scale, color=arrow_color, linewidth=line_width, connectionstyle=_connectionstyle, linestyle=linestyle, zorder=1)
            arrow_collection.append(arrow)
            ax.add_patch(arrow)
        return arrow_collection
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - minx
    h = maxy - miny
    if use_linecollection:
        edge_viz_obj = _draw_networkx_edges_line_collection()
        selfloops_to_draw = [loop for loop in nx.selfloop_edges(G) if loop in edgelist]
        if selfloops_to_draw:
            fancy_edges_indices = [edgelist_tuple.index(loop) for loop in selfloops_to_draw]
            edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in selfloops_to_draw])
            arrowstyle = '-'
            _draw_networkx_edges_fancy_arrow_patch()
    else:
        fancy_edges_indices = range(len(edgelist))
        edge_viz_obj = _draw_networkx_edges_fancy_arrow_patch()
    padx, pady = (0.05 * w, 0.05 * h)
    corners = ((minx - padx, miny - pady), (maxx + padx, maxy + pady))
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return edge_viz_obj