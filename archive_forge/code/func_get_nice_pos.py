import itertools
import functools
import importlib.util
def get_nice_pos(G, *, dim=2, layout='auto', initial_layout='auto', iterations='auto', k=None, use_forceatlas2=False, flatten=False, **layout_opts):
    if layout == 'auto' and HAS_PYGRAPHVIZ:
        layout = 'neato'
    if layout in ('dot', 'neato', 'fdp', 'sfdp'):
        pos = layout_pygraphviz(G, prog=layout, dim=dim)
        if layout != 'dot':
            pos = massage_pos(pos, flatten=flatten)
        return pos
    import networkx as nx
    if layout != 'auto':
        initial_layout = layout
        iterations = 0
    if initial_layout == 'auto':
        if len(G) <= 100:
            initial_layout = 'kamada_kawai'
        else:
            initial_layout = 'spectral'
    if iterations == 'auto':
        iterations = max(200, 1000 - len(G))
    if dim == 2.5:
        dim = 3
        project_back_to_2d = True
    else:
        project_back_to_2d = False
    if dim != 2:
        layout_opts['dim'] = dim
    pos0 = getattr(nx, initial_layout + '_layout')(G, **layout_opts)
    if iterations:
        if use_forceatlas2 is True:
            use_forceatlas2 = 1
        elif use_forceatlas2 in (0, False):
            use_forceatlas2 = float('inf')
        should_use_fa2 = HAS_FA2 and len(G) > use_forceatlas2 and (dim == 2)
        if should_use_fa2:
            from fa2 import ForceAtlas2
            pos = ForceAtlas2(verbose=False).forceatlas2_networkx_layout(G, pos=pos0, iterations=iterations)
        else:
            pos = nx.spring_layout(G, pos=pos0, k=k, dim=dim, iterations=iterations)
    else:
        pos = pos0
    if project_back_to_2d:
        pos = {k: v[:2] for k, v in pos.items()}
        dim = 2
    if dim == 2:
        pos = massage_pos(pos)
    return pos